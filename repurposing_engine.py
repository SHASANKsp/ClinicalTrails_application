import pandas as pd


class RepurposingSignalEngine:

    def __init__(self, graph):
        self.graph = graph

    def fetch_data(self, drug_name):

        query = """
        MATCH (i:Intervention {name: $drug})<-[:USES_INTERVENTION]-(s:Study)
        OPTIONAL MATCH (s)-[:STUDIES]->(c:Condition)
        OPTIONAL MATCH (sp:Sponsor)-[:SPONSORS]->(s)
        OPTIONAL MATCH (s)-[:CONDUCTED_AT]->(l:Location)
        OPTIONAL MATCH (s)-[:HAS_ARM]->(a:Arm)
        RETURN
            s.nct_id AS nct_id,
            s.phases AS phase,
            s.study_type AS study_type,
            s.allocation AS allocation,
            s.masking AS masking,
            s.overall_status AS status,
            s.enrollment AS enrollment,
            c.name AS condition,
            sp.name AS sponsor,
            l.country AS country,
            a.label AS arm_label
        """

        result = self.graph.query(query, params={"drug": drug_name})

        if not result:
            return None

        return pd.DataFrame(result)