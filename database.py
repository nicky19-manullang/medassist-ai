from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "mysql://root:AOYkwnNqzkfDjGDMSKVpbQOWaYCWqJiZ@mysql.railway.internal:3306/railway"
# DATABASE_URL = "mysql+pymysql://root:@localhost:3307/medassist_ai"


engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
