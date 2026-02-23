import hashlib
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, TIMESTAMP
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

DATABASE_URL = "sqlite:///chat_memory.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    phone = Column(String(15), index=True)
    role = Column(String(20))
    content = Column(Text)
    escalation = Column(Boolean, default=False)
    guardrail_triggered = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

Base.metadata.create_all(bind=engine)


def detect_flags(response_text: str):
    escalation = "Escalated to human support" in response_text
    guardrail = "outside the supported scope" in response_text
    return escalation, guardrail


def append_user_history(phone, message):
    """
    message format expected:
    {
        "query": "...",
        "response": "..."
    }
    """

    db = SessionLocal()

    # Store user message
    db.add(Message(
        phone=phone,
        role="user",
        content=message["query"]
    ))

    # Detect flags
    escalation, guardrail = detect_flags(message["response"])

    # Store assistant message
    db.add(Message(
        phone=phone,
        role="assistant",
        content=message["response"],
        escalation=escalation,
        guardrail_triggered=guardrail
    ))

    db.commit()
    db.close()


def get_user_history(phone):
    db = SessionLocal()

    messages = (
        db.query(Message)
        .filter(Message.phone == phone)
        .order_by(Message.created_at)
        .all()
    )

    history = [
        {
            "role": msg.role,
            "content": msg.content
        }
        for msg in messages
    ]

    db.close()
    return history