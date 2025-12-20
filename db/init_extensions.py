from db.connection import get_db_connection

def init_pgvector():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Initialize any required extensions here
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

if __name__ == "__main__":
    init_pgvector()
    print("pgvector extension ready")