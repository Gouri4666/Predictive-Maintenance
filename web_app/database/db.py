import os
import pymysql
import pymysql.cursors
from config import Config

def get_db_connection(use_database=True):
    """
    Establish and return a connection to MySQL database using PyMySQL.
    """
    try:
        conn = pymysql.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            port=Config.MYSQL_PORT,
            database=Config.MYSQL_DB if use_database else None,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
            connect_timeout=5
        )
        return conn
    except pymysql.MySQLError as err:
        print(f"[Database Error] Connection failed: {err}")
        raise err

def init_db():
    """
    Initialize database schema: create database if missing and apply schema.sql.
    """
    print("[Database] Initializing MySQL database...")
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")
    
    # 1. Connect to MySQL server without database to create DB if needed
    try:
        server_conn = get_db_connection(use_database=False)
        with server_conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{Config.MYSQL_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        server_conn.close()
        print(f"[Database] Verified/Created database: {Config.MYSQL_DB}")
    except Exception as e:
        print(f"[Database Warning] Could not verify/create DB: {e}")
        raise e

    # 2. Connect to the specific database and execute schema statements
    conn = get_db_connection(use_database=True)
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()

        # Split commands by semicolon while respecting comments and line breaks
        statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]
        
        with conn.cursor() as cur:
            for statement in statements:
                # Skip standalone USE or empty statements
                if statement.upper().startswith('USE ') or statement.upper().startswith('CREATE DATABASE'):
                    continue
                try:
                    cur.execute(statement)
                except Exception as stmt_err:
                    print(f"[Database Warning] Statement warning/skip: {stmt_err}")

            # Ensure columns last_inspection_at and next_inspection_due exist on machines table
            try:
                cur.execute("ALTER TABLE machines ADD COLUMN last_inspection_at DATETIME NULL DEFAULT NULL;")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE machines ADD COLUMN next_inspection_due DATETIME NULL DEFAULT NULL;")
            except Exception:
                pass

        print("[Database] Schema and initial seed data applied successfully.")
    finally:
        conn.close()

def query_db(query, args=None, one=False):
    """
    Execute a SELECT query and return rows as dictionary.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if args:
                cur.execute(query, args)
            else:
                cur.execute(query)
            results = cur.fetchall()
            return (results[0] if results else None) if one else results
    finally:
        conn.close()

def modify_db(query, args=None):
    """
    Execute an INSERT, UPDATE, or DELETE query and return lastrowid / rowcount.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if args:
                cur.execute(query, args)
            else:
                cur.execute(query)
            return cur.lastrowid
    finally:
        conn.close()
