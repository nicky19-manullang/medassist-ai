import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# local
# DATABASE_URL = "mysql+pymysql://root:@localhost:3307/medassist_ai"

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set!")

# Tambahkan driver jika Railway masih mysql://
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()