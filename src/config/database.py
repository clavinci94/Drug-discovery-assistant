import sqlite3, os
DATABASE_PATH = 'data/drug_discovery.db'

def setup_database():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH); c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS molecules (
            id INTEGER PRIMARY KEY,
            chembl_id VARCHAR(20) UNIQUE,
            smiles TEXT NOT NULL,
            molecular_weight REAL,
            logp REAL,
            hbd INTEGER,
            hba INTEGER,
            tpsa REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS targets (
            id INTEGER PRIMARY KEY,
            uniprot_id VARCHAR(20) UNIQUE,
            target_name TEXT,
            organism TEXT,
            protein_sequence TEXT,
            target_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS bioactivities (
            id INTEGER PRIMARY KEY,
            molecule_id INTEGER,
            target_id INTEGER,
            activity_type VARCHAR(20),
            activity_value REAL,
            activity_unit VARCHAR(10),
            assay_type VARCHAR(50),
            confidence_score INTEGER,
            FOREIGN KEY (molecule_id) REFERENCES molecules(id),
            FOREIGN KEY (target_id) REFERENCES targets(id)
        )
    ''')
    conn.commit(); conn.close()
    print(f"✅ Database setup complete: {DATABASE_PATH}")

if __name__ == "__main__":
    setup_database()
