from datetime import datetime
from uuid import uuid4


def create_escalation_case_id():
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    short_uuid = str(uuid4()).split("-")[0].upper()
    return f"ESC-{timestamp}-{short_uuid}"


def simulate_human_handoff(name, phone, email, issue):
    case_id = create_escalation_case_id()

    return {
        "case_id": case_id,
        "email_status": f"Email queued to {email}",
        "whatsapp_status": f"WhatsApp notification queued to {phone}",
        "summary": f"Escalation logged for {name}: {issue}"
    }


def human_escalation_form(name, phone, email, issue, case_id=None):
    ticket = case_id or "Pending"

    return f"""
    Thank you {name}.
    We have received your issue.
    Phone: {phone}
    Email: {email}

    Our human support team will contact you shortly.
    """
