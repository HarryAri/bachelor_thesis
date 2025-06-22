from config.chat import chat
from neo4j import GraphDatabase

npc_id = "N004"
knowledge_graph = GraphDatabase.driver(
    "bolt://localhost:7687", auth=(" ", " ")
)

def get_all_related_nodes(npc_id):
    '''
    :param npc_id: the npc you want to have conversation with
    :return: the 1st degree nodes connected to NPC
    '''
    query = """
    MATCH (npc:NPC {id: $npc_id}) -[r]- (e)
    RETURN type(r) AS relation, labels(e) AS labels, e.name AS name, e.description AS description
    """
    with knowledge_graph.session() as session:
        result = session.run(query, npc_id=npc_id)
        related_nodes = []
        for record in result:
            name = record["name"]
            desc = record["description"]
            rel = record["relation"]
            label = ", ".join(record["labels"]) if record["labels"] else "Entity"
            related_nodes.append({
                "name": f"{name} ({label}, {rel})",
                "description": desc
            })
    return related_nodes

def run_kg(question, llm):
    '''
    :param question: input query from the user
    :param llm: the LLM used to generate the response
    :return: NPC's response to the user query
    '''
    facts = get_all_related_nodes(npc_id)
    retrieved_info = "\n".join(f"- {f['name']}: {f['description']}" for f in facts)
    return chat(llm, question, retrieved_info)