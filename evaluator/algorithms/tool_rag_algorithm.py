import json
import math
import os
import re
import unicodedata
from itertools import chain
from json import JSONDecodeError

import numpy as np

from typing import List, Dict, Tuple, Any
from langchain.docstore.document import Document
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_milvus import Milvus, BM25BuiltInFunction
from langgraph.prebuilt import create_react_agent
from pymilvus import connections, utility

from evaluator.components.data_provider import QuerySpecification
from evaluator.components.llm_provider import query_llm
from evaluator.config.defaults import VERBOSE
from evaluator.config.schema import ModelConfig
from evaluator.utils.module_extractor import register_algorithm
from evaluator.interfaces.algorithm import Algorithm, AlgoResponse

from dotenv import load_dotenv

from evaluator.utils.utils import print_verbose

load_dotenv()

_CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_WS_RE = re.compile(r'\s+')


MILVUS_CONNECTION_ALIAS = "tools_connection"
COLLECTION_NAME = "tools_collection"
OVERRIDE_COLLECTION = True


if not VERBOSE:
    # Silence gRPC C-core and tracing
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_TRACE"] = ""
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


class L2Wrapper:
    def __init__(self, base):
        self.base = base

    def embed_documents(self, texts):
        x = np.array(self.base.embed_documents(texts), dtype=np.float32)
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        return x.tolist()

    def embed_query(self, text):
        x = np.array(self.base.embed_query(text), dtype=np.float32)
        x /= (np.linalg.norm(x) + 1e-12)
        return x.tolist()


@register_algorithm("tool_rag")
class ToolRagAlgorithm(Algorithm):
    """
    Optional configurable settings (to be provided in the 'settings' dictionary):

    - top_k: the number of documents to retrieve on a given query.
    - embedding_model_id: the ID of the model to use for embedding calculation.
    - similarity_metric: the metric to use for embedding distance calculation - can be COSINE, L2 or IP (inner product).
    - index_type: the type of Milvus index to build - can be FLAT, HNSW, IVF_FLAT or any other supported type.
    - tau: an optional threshold between 0 and 1 specifying the maximal allowed distance to the query. All tools having
      greater distance will be filtered. Set to None to disable this filter.
    - sim_threshold: an optional threshold between 0 and 1 specifying the maximal allowed similarity between the retrieved
      tools. Sets of highly similar tools will be filtered to only retain one. Set to None to disable this filter.
    - text_preprocessing_operations: a list of preprocessing operations to apply on the indexed documents and the
      query text. The following operations are supported: 'unicode_normalization', 'lowercase', 'collapse_whitespaces',
      'split_camel_snake_case'. Set this parameter to None to disable preprocessing or to 'all' to enable all operations.
    - max_document_size: the maximal size, in characters, of a single indexed document, or None to disable the size limit.
    - indexed_tool_def_parts: the parts of the MCP tool definition to be used for index construction, such as 'name',
      'description', 'args', etc.
    - hybrid_mode: True to enable hybrid (sparse + dense) search and False to only enable dense search.
    - analyzer_params: parameters for the Milvus BM25 analyzer.
    - fusion_type: the algorithm for combining the dense and the sparse scores if hybrid mode is activated. Milvus only
      supports "weighted" and "rrf".
    - fusion_alpha: the relative weight of the dense retriever score. The final score is calculated as
      alpha*dense + (1-alpha)*sparse. This parameter is only used with the "weighted" fusion type.
    - fusion_k: the k value to use for the "rrf" hybrid fusion mode.
    - cross_encoder_model_name: the name of the model to use for reranking or None to disable reranking.
    - reranker_pool_size: the number of results to retrieve from the vector DB before reranking. Must be greater than or
      equal to top_k.
    - enable_query_decomposition: True to enable query decomposition into subtasks and False otherwise.
    - enable_query_rewriting: True to enable (sub-)query rewriting into a list of relevant APIs and False otherwise.
    - query_rewriting_model_id: the ID of the model to use for query rewriting and decomposition.
    - min_sub_tasks: the minimum number of tasks to decompose the original query into.
    - max_sub_tasks: the maximum number of tasks to decompose the original query into.
    - query_rewrite_tool_suggestions_num: the maximal number of tool APIs to produce from the original query during rewriting.
    """

    model: BaseChatModel or None
    vector_store: Milvus or None
    reranker: CrossEncoderReranker or None
    query_rewriting_model: BaseChatModel or None
    embeddings: HuggingFaceEmbeddings or None

    # due to the limitations of Langchain tools we cannot truly serialize them. Therefore, indexing
    # the tools themselves is not possible. Instead, we keep all tools in memory and only index their unique IDs (names)
    # which are later used to retrieve the actual tools.
    tool_name_to_base_tool: Dict[str, BaseTool] or None

    def __init__(self, settings: Dict, model_config: List[ModelConfig], label: str = None):
        super().__init__(settings, model_config, label)

        self.model = None
        self.tool_name_to_base_tool = None
        self.vector_store = None
        self.reranker = None
        self.query_rewriting_model = None
        self.embeddings = None

    def get_default_settings(self) -> Dict[str, Any]:
        return {
            # basic
            "top_k": 10,
            "embedding_model_id": "all-MiniLM-L6-v2",
            "similarity_metric": "COSINE",
            "index_type": "FLAT",
            "indexed_tool_def_parts": ["name", "description"],

            # preprocessing
            "text_preprocessing_operations": None,
            "max_document_size": None,

            # hybrid search
            "hybrid_mode": False,
            "analyzer_params": None,
            "fusion_type": "rrf",
            "fusion_k": 100,
            "fusion_alpha": 0.5,

            # reranking
            "cross_encoder_model_name": None,
            "reranker_pool_size": 50,

            # query rewriting / decomposition
            "enable_query_decomposition": False,
            "enable_query_rewriting": False,
            "query_rewriting_model_id": "llama32-3b",
            "min_sub_tasks": 1,
            "max_sub_tasks": 5,
            "query_rewrite_tool_suggestions_num": 3,

            # post-retrieval filtering
            "tau": None,
            "sim_threshold": None,
        }

    def _preprocess_text(self, text: str) -> str:
        ops = self._settings["text_preprocessing_operations"]
        if not ops:
            return text

        t = text
        if "unicode_normalization" in ops or "all" in ops:
            t = unicodedata.normalize('NFC', t)
        if "lowercase" in ops or "all" in ops:
            t = t.lower()
        if "collapse_whitespaces" in ops or "all" in ops:
            t = _WS_RE.sub(' ', t).strip()
        if "split_camel_snake_case" in ops or "all" in ops:
            t = t.replace('_', ' ')  # snake_case -> snake case
            t = _CAMEL_RE.sub(' ', t)  # camelCase -> camel Case

        return t

    def _truncate(self, s: str, sep: str = " ... ") -> str:
        max_chars = self._settings["max_document_size"]
        if max_chars is None or len(s) <= max_chars:
            return s

        # Performs a simple head-tail truncation to max_chars characters.
        # We may consider supporting more sophisticated techniques, such as token-based truncation in the future.
        half = (max_chars - len(sep)) // 2
        return s[:half] + sep + s[-half:]

    @staticmethod
    def _render_args_schema(schema: Dict[str, Any]) -> str:
        if not schema:
            return ""
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        ordered = list(required) + [p for p in props.keys() if p not in required]
        # Keep it terse: "name:type" when available
        parts = []
        for name in ordered:
            spec = props.get(name) or {}
            t = spec.get("type")
            if isinstance(t, str):
                parts.append(f"{name}:{t}")
            else:
                parts.append(name)
        return " ".join(parts)

    @staticmethod
    def _render_examples(examples: List[str], max_examples: int = 3) -> str:
        exs = (examples or [])[:max_examples]
        return " || ".join(exs)

    def _compose_tool_text(self, tool: BaseTool) -> str:
        parts_to_include = self._settings["indexed_tool_def_parts"]
        if not parts_to_include:
            raise ValueError("indexed_tool_def_parts must be a non-empty list")

        segments = []
        for p in parts_to_include:
            if p.lower() == "name":
                val = tool.name
                if val:
                    segments.append(f"name: {val}")
            elif p.lower() == "description":
                val = tool.description
                if val:
                    segments.append(f"desc: {val}")
            elif p.lower() == "args":
                val = self._render_args_schema(tool.args_schema)
                if val:
                    segments.append(f"args: {val}")
            elif p.lower() == "tags":
                tags = tool.tags or []
                if tags:
                    segments.append(f"tags: {' '.join(tags)}")

        if not segments:
            raise ValueError(f"The following tool contains none of the fields listed in indexed_tool_def_parts:\n{tool}")
        text = " | ".join(segments)

        # one-pass preprocess + truncation
        text = self._preprocess_text(text)
        text = self._truncate(text)
        return text

    def _create_docs_from_tools(self, tools: List[BaseTool]) -> List[Document]:
        documents = []
        for tool in tools:
            page_content = self._compose_tool_text(tool)
            documents.append(Document(page_content=page_content, metadata={"name": tool.name}))
        return documents

    def _index_tools(self, tools: List[BaseTool]) -> None:
        self.tool_name_to_base_tool = {tool.name: tool for tool in tools}

        self.embeddings = HuggingFaceEmbeddings(model_name=self._settings["embedding_model_id"])
        if self._settings["similarity_metric"] == "COSINE":
            # L2-normalizing embedding vectors before cosine similarity makes the results more stable
            embeddings = L2Wrapper(self.embeddings)
        else:
            embeddings = self.embeddings

        milvus_uri = os.getenv("MILVUS_PATH")

        index_params = {
            "index_type": self._settings["index_type"],
            "metric_type": self._settings["similarity_metric"],
        }

        search_params = {
            "metric_type": self._settings["similarity_metric"],
        }

        connections.connect(alias=MILVUS_CONNECTION_ALIAS, uri=milvus_uri)
        if not OVERRIDE_COLLECTION and utility.has_collection(COLLECTION_NAME):
            print_verbose(f"Loading Milvus server collection: {COLLECTION_NAME}")
            self.vector_store = Milvus(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": milvus_uri},
                index_params=index_params,
                search_params=search_params,
            )

        print_verbose(f"Creating new Milvus collection on the server: {COLLECTION_NAME}")
        if self._settings["hybrid_mode"]:
            # index and search parameters must be extended for sparse search
            index_params = [index_params, None]
            search_params = [search_params, {}]
            self.vector_store = Milvus.from_documents(
                documents=self._create_docs_from_tools(tools),
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": milvus_uri},
                drop_old=True,
                index_params=index_params,
                search_params=search_params,
                builtin_function=BM25BuiltInFunction(analyzer_params=self._settings["analyzer_params"]),
                vector_field=["dense", "sparse"],
            )
        else:
            self.vector_store = Milvus.from_documents(
                documents=self._create_docs_from_tools(tools),
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": milvus_uri},
                drop_old=True,
                index_params=index_params,
                search_params=search_params,
            )

    def set_up(self, model: BaseChatModel, tools: List[BaseTool]) -> None:
        self.model = model

        if self._settings["cross_encoder_model_name"]:
            self.reranker = CrossEncoderReranker(
                model=HuggingFaceCrossEncoder(model_name=self._settings["cross_encoder_model_name"]),
                top_n=self._settings["top_k"],
            )

        if self._settings["enable_query_decomposition"] or self._settings["enable_query_rewriting"]:
            self.query_rewriting_model = self._get_llm(self._settings["query_rewriting_model_id"])

        self._index_tools(tools)

    def _threshold_results(self, docs_and_scores: List[Tuple[Document, float]]) -> List[Document]:
        """
        Filters search results according to the tau parameter to only retain those similar enough to the query.
        """
        tau = self._settings["tau"]
        if tau is None:
            # tau not specified - skip post-search filtering
            return [d for (d, s) in docs_and_scores]

        metric = self._settings["similarity_metric"]

        if metric.upper() in {"COSINE", "IP"}:
            kept = [(d, s) for (d, s) in docs_and_scores if s >= tau]
            kept.sort(key=lambda x: x[1], reverse=True)  # higher is better
        elif metric.upper() == "L2":
            kept = [(d, s) for (d, s) in docs_and_scores if s <= tau]
            kept.sort(key=lambda x: x[1])  # lower is better
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return [d for (d, s) in kept]

    def _embed_docs(self, docs: List[Document]) -> np.ndarray:
        """Encode docs -> (n, d), then L2-normalize for cosine."""
        texts = [d.page_content for d in docs]
        vecs = np.asarray(self.embeddings.embed_documents(texts), dtype=np.float32)
        # L2-normalize the vectors
        n = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(n, 1e-12)

    def _diversify_nms(self, docs: List[Document]) -> List[Document]:
        """
        Filters the documents by rank-preserving diversity (NMS).
        """
        if not docs:
            return []

        embedded = self._embed_docs(docs)  # (n,d), L2-normalized
        kept_indices = []
        for i in range(len(docs)):
            v = embedded[i]
            # if nothing kept yet, keep the first one (best by reranker)
            if not kept_indices:
                kept_indices.append(i)
                continue
            # compute similarity to all kept
            sims = (embedded[kept_indices] @ v)  # (len(kept),)
            if float(np.max(sims)) < self._settings["sim_threshold"]:
                kept_indices.append(i)

        return [docs[i] for i in kept_indices]

    def _postprocess_results(self, docs_and_scores: List[Tuple[Document, float]], query: str) -> List[Document]:
        """
        Optionally filters and/or reranks results from the vector DB search.
        """
        if self._settings["tau"] is not None:
            # tau is specified - thresholding is enabled
            docs = self._threshold_results(docs_and_scores)
        else:
            docs = [d for (d, s) in docs_and_scores]

        if self._settings["cross_encoder_model_name"] is not None:
            # reranking is enabled
            docs = self.reranker.compress_documents(docs, query)

        if self._settings["sim_threshold"] is not None:
            # NMS-based diversity filtering is enabled
            docs = self._diversify_nms(docs)

        return docs

    def _decompose_query(self, query: str) -> List[str]:
        """
        Decomposes a given, possibly highly complex query into a list of simpler subtasks.
        """
        system_prompt = "You are a helpful assistant."

        user_prompt = (
            "Split the USER REQUEST provided below into atomic, simple subtasks. "
            "Each subtask should map to a single API/tool call. "
            f"Produce between {self._settings['min_sub_tasks']} and {self._settings['max_sub_tasks']} subtasks. "
            f"Subtasks should be written in imperative style and contain at most 25 words."
            f"Merge near-duplicates. "
            f"Skip steps that do not require a tool. "
            f"Return JSON: {{\"subtasks\": [\"...\"]}}"
            f"USER REQUEST:\n {query}\n"
        )

        raw = query_llm(self.query_rewriting_model, system_prompt, user_prompt)
        parsed = self._safe_json_parse(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("subtasks"), list):
            items = [str(x) for x in parsed["subtasks"]]
        else:
            # fallback to line-by-line if JSON wasn't returned
            items = self._lines(raw)

        steps: List[str] = []
        for s in items:
            s = self._strip_numbering(s)
            if not s:
                continue
            steps.append(s)
            if len(steps) >= self._settings['max_sub_tasks']:
                break

        steps = self._dedup_keep_order(steps)
        if not steps:
            steps = [query]

        return steps

    def _rewrite_query_to_tool_descriptions(self, query: str) -> List[str]:
        """
        Transforms a query into a list of hypothetical tool APIs required to complete it.
        """
        system_prompt = "You are a helpful assistant."

        num_tools = self._settings['query_rewrite_tool_suggestions_num']
        user_prompt = (
            f"Generate short descriptions of API(s) that can be used to address USER REQUEST as specified below. "
            f"Each description must be concise and phrased like a capability (not an answer). "
            f"Do NOT answer the user. "
            f"Write up to {num_tools} short descriptions. "
            f"Each description should contain at most 25 words. "
            f"Output one description per line."
            f"USER REQUEST:\n {query}\n"
        )

        raw = query_llm(self.query_rewriting_model, system_prompt, user_prompt)
        items = self._lines(raw)[:max(1, num_tools)]
        items = self._dedup_keep_order([s for s in items if s])

        return items

    @staticmethod
    def _merge_tool_docs(
            tool_docs_with_scores_lists: List[List[Tuple[Document, float]]]) -> List[Tuple[Document, float]]:
        merged = []

        for doc, score in chain.from_iterable(tool_docs_with_scores_lists):
            # Find if an equal document already exists in the merged list
            idx = next((i for i, (d, _) in enumerate(merged) if d.metadata["name"] == doc.metadata["name"]), None)

            if idx is None:
                merged.append((doc, score))
            else:
                # Keep the higher score
                if score > merged[idx][1]:
                    merged[idx] = (doc, score)

        merged.sort(key=lambda pair: pair[1], reverse=True)
        return merged

    async def _get_tools_for_sub_query(self, query: str, k: int) -> List[Tuple[Document, float]]:
        if not self._settings["enable_query_rewriting"]:
            # just fetch the tools from the DB based on the query text
            return await self._fetch_tools_for_sub_query_from_vector_db(query, k)

        tool_descriptions = self._rewrite_query_to_tool_descriptions(query)
        tool_candidates_per_tool_descriptions = math.ceil(max(k / len(tool_descriptions), 1))
        all_tool_lists = []
        for tool_description in tool_descriptions:
            current_tools = await self._fetch_tools_for_sub_query_from_vector_db(
                tool_description,
                tool_candidates_per_tool_descriptions,
            )
            all_tool_lists.append(current_tools)

        return self._merge_tool_docs(all_tool_lists)

    async def _fetch_tools_for_sub_query_from_vector_db(self, sub_query: str, k: int) -> List[Tuple[Document, float]]:
        query_text = self._preprocess_text(sub_query)

        if self._settings["hybrid_mode"]:
            hybrid_fusion_type = self._settings["fusion_type"]
            if hybrid_fusion_type == "weighted":
                alpha = self._settings["fusion_alpha"]
                hybrid_fusion_params = {"weights": [alpha, 1 - alpha]}
            elif hybrid_fusion_type == "rrf":
                hybrid_fusion_params = {"k": self._settings["fusion_k"]}
            else:
                raise ValueError(f"Unsupported hybrid fusion type: {hybrid_fusion_type}")
        else:
            hybrid_fusion_type = None
            hybrid_fusion_params = None

        return self.vector_store.similarity_search_with_score(
            query_text,
            k=k,
            ranker_type=hybrid_fusion_type,
            ranker_params=hybrid_fusion_params,
        )

    async def _get_tools_for_query(self, query: str) -> List[Tuple[Document, float]]:

        # if reranking is enabled, we have to retrieve more results than without reranking
        total_k = self._settings["reranker_pool_size"] if self.reranker is not None else self._settings["top_k"]

        if not self._settings["enable_query_decomposition"]:
            # there is just one sub-query
            return await self._get_tools_for_sub_query(query, total_k)

        sub_queries = self._decompose_query(query)
        tools_per_sub_query = math.ceil(max(total_k / len(sub_queries), 1))
        all_tool_lists = []
        for sub_q in sub_queries:
            current_tools = await self._get_tools_for_sub_query(sub_q, tools_per_sub_query)
            all_tool_lists.append(current_tools)

        return self._merge_tool_docs(all_tool_lists)

    async def process_query(self, query_spec: QuerySpecification) -> AlgoResponse:
        if not self.vector_store:
            raise RuntimeError("process_query called before set_up")

        print_verbose(f"Retrieving documents for query: {query_spec.query}")
        docs_and_scores = await self._get_tools_for_query(query_spec.query)

        relevant_documents = self._postprocess_results(docs_and_scores, self._preprocess_text(query_spec.query))
        relevant_tool_names = [d.metadata["name"] for d in relevant_documents]

        print_verbose(f"Retrieved tools for query #{query_spec.id}: {relevant_tool_names}")
        relevant_tools = [self.tool_name_to_base_tool[name] for name in relevant_tool_names]

        agent = create_react_agent(self.model, relevant_tools)
        print_mode = "debug" if VERBOSE else ()
        response = await agent.ainvoke(
            input={"messages": query_spec.query},
            max_iterations=6,
            print_mode=print_mode
        )
        return response, relevant_tool_names

    def tear_down(self) -> None:
        connections.disconnect(alias=MILVUS_CONNECTION_ALIAS)

    @staticmethod
    def _safe_json_parse(text: str):
        text = (text or "").strip()
        try:
            return json.loads(text)
        except JSONDecodeError:
            m = re.search(r"(\{.*}|\[.*])", text, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _lines(text: str) -> List[str]:
        return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

    @staticmethod
    def _dedup_keep_order(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    @staticmethod
    def _strip_numbering(s: str) -> str:
        return re.sub(r"^\s*(?:[-*]|\d+[).:]?)\s*", "", s).strip().rstrip(".")
