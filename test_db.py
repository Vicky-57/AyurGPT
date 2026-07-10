import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    host = os.getenv("SUPABASE_DB_HOST")
    database = os.getenv("SUPABASE_DB_NAME", "postgres")
    user = os.getenv("SUPABASE_DB_USER", "postgres")
    password = os.getenv("SUPABASE_DB_PASS")
    port = os.getenv("SUPABASE_DB_PORT", "5432")
    
    print("Connecting to Supabase PostgreSQL database...")
    print(f"Host: {host}")
    print(f"Database: {database}")
    print(f"User: {user}")
    print(f"Port: {port}")
    
    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port,
            connect_timeout=10
        )
        print("[OK] Successfully connected to Supabase!")
        
        # Enable autocommit
        conn.autocommit = True
        cur = conn.cursor()
        
        # 1. Create table
        print("Creating 'chat_history' table if it does not exist...")
        create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            sources JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)
        print("Table 'chat_history' verified/created.")
        
        # 2. Create index
        print("Creating index on 'session_id'...")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id);")
        print("Index verified/created.")
        
        # 3. Test insert
        print("Inserting dummy test record...")
        test_session_id = "test-session-1234"
        test_role = "user"
        test_content = "Namaste, this is a database connection test."
        test_sources = json.dumps(["Page 1", "Page 2"])
        
        cur.execute(
            "INSERT INTO chat_history (session_id, role, content, sources) VALUES (%s, %s, %s, %s) RETURNING id;",
            (test_session_id, test_role, test_content, test_sources)
        )
        inserted_id = cur.fetchone()[0]
        print(f"Inserted record with ID: {inserted_id}")
        
        # 4. Test fetch
        print("Fetching test record back...")
        cur.close()
        dict_cur = conn.cursor(cursor_factory=RealDictCursor)
        dict_cur.execute(
            "SELECT * FROM chat_history WHERE session_id = %s ORDER BY created_at ASC;",
            (test_session_id,)
        )
        records = dict_cur.fetchall()
        print(f"Fetched {len(records)} records:")
        for r in records:
            print(f"  [{r['role']}] {r['content']} (Sources: {r['sources']})")
            
        # 5. Clean up test record
        print("Cleaning up test record...")
        dict_cur.execute("DELETE FROM chat_history WHERE session_id = %s;", (test_session_id,))
        print("Cleanup completed.")
        
        dict_cur.close()
        conn.close()
        print("[OK] Database test completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Database test failed: {e}")

if __name__ == "__main__":
    main()
