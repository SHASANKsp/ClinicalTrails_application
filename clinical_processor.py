import re
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.graphs import Neo4jGraph


class ClinicalGraphRAGProcessor:

    def __init__(self, graph: Neo4jGraph, llm: OllamaLLM):
        self.graph = graph
        self.llm = llm
        self.schema_text = self.build_schema_description()

    # -------------------------------------
    # STEP 0: Automatic Schema Introspection
    # -------------------------------------

    def build_schema_description(self):

        try:
            result = self.graph.query("CALL db.schema.visualization()")
        except Exception as e:
            return f"Schema unavailable: {str(e)}"

        if not result:
            return "Schema unavailable."

        schema_summary = ""

        record = result[0]

        nodes = record.get("nodes", [])
        relationships = record.get("relationships", [])

        schema_summary += "Node Labels:\n"
        for node in nodes:
            # Some Neo4j versions return dict, some return object
            if isinstance(node, dict):
                schema_summary += f"- {node.get('name', node)}\n"
            else:
                schema_summary += f"- {node}\n"

        schema_summary += "\nRelationships:\n"

        for rel in relationships:

            # If tuple format: (start, type, end)
            if isinstance(rel, tuple) and len(rel) == 3:
                start, rel_type, end = rel
                schema_summary += f"- ({start})-[:{rel_type}]->({end})\n"

            # If dict format
            elif isinstance(rel, dict):
                schema_summary += (
                    f"- ({rel.get('startNode')})"
                    f"-[:{rel.get('type')}]->"
                    f"({rel.get('endNode')})\n"
                )

            else:
                schema_summary += f"- {rel}\n"

        return schema_summary

    # -------------------------------------
    # STEP 1: Generate Cypher
    # -------------------------------------

    def generate_cypher(self, question: str):

        system_prompt = """
        You are a Neo4j Cypher expert.
        Generate ONLY a valid Cypher query.
        Do NOT use markdown formatting.
        Do NOT wrap in backticks.
        Use ONLY labels and relationships present in the schema.
        If unsure about entity type, prefer Intervention for drug names.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Graph Schema:
            {self.schema_text}
            
            User Question:
            {question}
            
            Generate Cypher:
            """)
        ])

        cypher = self.llm.invoke(prompt.format_messages())
        return self.clean_cypher(cypher)

    # -------------------------------------
    # STEP 2: Clean Cypher
    # -------------------------------------

    def clean_cypher(self, cypher: str):

        cypher = cypher.strip()

        # Remove markdown fences
        cypher = re.sub(r"```[a-zA-Z]*", "", cypher)
        cypher = cypher.replace("```", "")

        # Remove accidental leading text
        cypher = cypher.replace("cypher", "", 1).strip()

        # Enforce read-only safety
        forbidden = ["CREATE", "MERGE", "DELETE", "DROP", "SET", "DETACH"]
        for word in forbidden:
            if word in cypher.upper():
                raise ValueError("Unsafe query detected.")

        return cypher

    # -------------------------------------
    # STEP 3: Execute Query
    # -------------------------------------

    def execute_query(self, cypher: str):

        try:
            return self.graph.query(cypher)
        except Exception as e:
            return {"error": str(e)}

    # -------------------------------------
    # STEP 4: Generate Grounded Answer
    # -------------------------------------

    def generate_answer(self, question: str, results):

        system_prompt = """
        You are a clinical research intelligence assistant.
        Answer ONLY using the provided graph results.
        Be concise, scientific, and precise.
        If data is insufficient, state that clearly.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            User Question:
            {question}
            
            Graph Results:
            {results}
            
            Provide a scientific answer:
            """)
        ])

        return self.llm.invoke(prompt.format_messages())

    # -------------------------------------
    # MAIN PIPELINE
    # -------------------------------------

    def process_query(self, question: str):

        steps = {}

        # Generate Cypher
        cypher = self.generate_cypher(question)
        steps["generated_cypher"] = cypher

        # Execute
        results = self.execute_query(cypher)
        steps["results"] = results

        if isinstance(results, dict) and "error" in results:
            return {
                "success": False,
                "steps": steps,
                "error": results["error"]
            }

        # Generate answer
        answer = self.generate_answer(question, results)
        steps["final_answer"] = answer

        return {
            "success": True,
            "steps": steps
        }