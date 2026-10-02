import os
from typing import Dict, Any
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

# For docker-compose we connect to 'db' host, fallback to localhost
DB_HOST = os.environ.get("POSTGRES_HOST", "db")
DATABASE_URL = f"postgresql://admin:password@{DB_HOST}:5432/fraud_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    time = Column(Float)
    amount = Column(Float)
    probability = Column(Float)
    decision = Column(Integer)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"Warning: Could not initialize DB (is Postgres running?): {e}")

def save_prediction(tx: Dict[str, Any], prob: float, pred: int):
    session = SessionLocal()
    try:
        db_pred = Prediction(
            time=tx.get("Time", 0.0),
            amount=tx.get("Amount", 0.0),
            probability=prob,
            decision=pred
        )
        session.add(db_pred)
        session.commit()
    except Exception as e:
        print(f"Error saving to DB: {e}")
        session.rollback()
    finally:
        session.close()

init_db()
