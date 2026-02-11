"""
ShopUNow Database Manager
=========================
Handles user authentication and conversation history storage
using SQLite database.

Tables:
- users: Store user information (mobile, name, created_at)
- conversations: Store all conversation history
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import hashlib
import re

class DatabaseManager:
    def __init__(self, db_path: str = "shopunow_assistant.db"):
        """Initialize database connection and create tables"""
        self.db_path = db_path
        self.conn = None
        self.initialize_database()
    
    def get_connection(self):
        """Get database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def initialize_database(self):
        """Create tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mobile_number TEXT UNIQUE NOT NULL,
                name TEXT,
                email TEXT,
                user_type TEXT DEFAULT 'customer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_conversations INTEGER DEFAULT 0
            )
        """)
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                message_role TEXT NOT NULL,
                message_content TEXT NOT NULL,
                departments TEXT,
                sentiment TEXT,
                escalated BOOLEAN DEFAULT 0,
                has_attachments BOOLEAN DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Sessions table - track active sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT UNIQUE NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                message_count INTEGER DEFAULT 0,
                departments_used TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user_session 
            ON conversations(user_id, session_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_mobile 
            ON users(mobile_number)
        """)
        
        conn.commit()
        print("✓ Database initialized successfully")
    
    # ========================================================================
    # USER MANAGEMENT
    # ========================================================================
    
    def validate_mobile_number(self, mobile: str) -> Tuple[bool, str]:
        """Validate mobile number format"""
        # Remove spaces, dashes, and parentheses
        cleaned = re.sub(r'[\s\-\(\)]', '', mobile)
        
        # Check if it's a valid format (10-15 digits, optional + at start)
        if re.match(r'^\+?\d{10,15}$', cleaned):
            return True, cleaned
        else:
            return False, "Invalid mobile number format. Use 10-15 digits with optional +"
    
    def create_or_get_user(self, mobile_number: str, name: str = None, 
                          email: str = None, user_type: str = 'customer') -> Dict:
        """Create new user or get existing user by mobile number"""
        # Validate mobile number
        is_valid, result = self.validate_mobile_number(mobile_number)
        if not is_valid:
            raise ValueError(result)
        
        mobile_number = result
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("""
            SELECT * FROM users WHERE mobile_number = ?
        """, (mobile_number,))
        
        user = cursor.fetchone()
        
        if user:
            # Update last active
            cursor.execute("""
                UPDATE users SET last_active = CURRENT_TIMESTAMP
                WHERE mobile_number = ?
            """, (mobile_number,))
            conn.commit()
            
            return {
                'id': user['id'],
                'mobile_number': user['mobile_number'],
                'name': user['name'],
                'email': user['email'],
                'user_type': user['user_type'],
                'created_at': user['created_at'],
                'total_conversations': user['total_conversations'],
                'is_new': False
            }
        else:
            # Create new user
            cursor.execute("""
                INSERT INTO users (mobile_number, name, email, user_type)
                VALUES (?, ?, ?, ?)
            """, (mobile_number, name, email, user_type))
            conn.commit()
            
            user_id = cursor.lastrowid
            
            return {
                'id': user_id,
                'mobile_number': mobile_number,
                'name': name,
                'email': email,
                'user_type': user_type,
                'created_at': datetime.now().isoformat(),
                'total_conversations': 0,
                'is_new': True
            }
    
    def update_user_info(self, user_id: int, name: str = None, 
                        email: str = None, user_type: str = None):
        """Update user information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if name:
            updates.append("name = ?")
            params.append(name)
        if email:
            updates.append("email = ?")
            params.append(email)
        if user_type:
            updates.append("user_type = ?")
            params.append(user_type)
        
        if updates:
            params.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get user statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get total conversations
        cursor.execute("""
            SELECT COUNT(DISTINCT session_id) as total_sessions,
                   COUNT(*) as total_messages
            FROM conversations
            WHERE user_id = ?
        """, (user_id,))
        
        stats = cursor.fetchone()
        
        # Get departments used
        cursor.execute("""
            SELECT DISTINCT departments
            FROM conversations
            WHERE user_id = ? AND departments IS NOT NULL AND departments != ''
        """, (user_id,))
        
        dept_rows = cursor.fetchall()
        departments = set()
        for row in dept_rows:
            if row['departments']:
                depts = json.loads(row['departments'])
                departments.update(depts)
        
        # Get escalation count
        cursor.execute("""
            SELECT COUNT(*) as escalation_count
            FROM conversations
            WHERE user_id = ? AND escalated = 1
        """, (user_id,))
        
        escalations = cursor.fetchone()
        
        return {
            'total_sessions': stats['total_sessions'] if stats else 0,
            'total_messages': stats['total_messages'] if stats else 0,
            'departments_used': list(departments),
            'escalation_count': escalations['escalation_count'] if escalations else 0
        }
    
    # ========================================================================
    # CONVERSATION MANAGEMENT
    # ========================================================================
    
    def create_session(self, user_id: int, session_id: str):
        """Create a new session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (user_id, session_id)
            VALUES (?, ?)
        """, (user_id, session_id))
        conn.commit()
    
    def end_session(self, session_id: str, departments_used: List[str] = None):
        """End a session and update statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Update session end time
        cursor.execute("""
            UPDATE sessions 
            SET ended_at = CURRENT_TIMESTAMP,
                departments_used = ?
            WHERE session_id = ?
        """, (json.dumps(departments_used) if departments_used else None, session_id))
        
        # Update user's total conversations
        cursor.execute("""
            UPDATE users
            SET total_conversations = total_conversations + 1
            WHERE id = (SELECT user_id FROM sessions WHERE session_id = ?)
        """, (session_id,))
        
        conn.commit()
    
    def save_message(self, user_id: int, session_id: str, role: str, 
                    content: str, departments: List[str] = None, 
                    sentiment: str = None, escalated: bool = False,
                    has_attachments: bool = False):
        """Save a single message to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations 
            (user_id, session_id, message_role, message_content, 
             departments, sentiment, escalated, has_attachments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, 
            session_id, 
            role, 
            content,
            json.dumps(departments) if departments else None,
            sentiment,
            escalated,
            has_attachments
        ))
        
        # Update session message count
        cursor.execute("""
            UPDATE sessions
            SET message_count = message_count + 1
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
    
    def get_conversation_history(self, user_id: int, 
                                 limit: int = 50) -> List[Dict]:
        """Get conversation history for a user (most recent first)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                c.*,
                s.started_at as session_started
            FROM conversations c
            LEFT JOIN sessions s ON c.session_id = s.session_id
            WHERE c.user_id = ?
            ORDER BY c.timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'role': row['message_role'],
                'content': row['message_content'],
                'session_id': row['session_id'],
                'departments': json.loads(row['departments']) if row['departments'] else [],
                'sentiment': row['sentiment'],
                'escalated': bool(row['escalated']),
                'has_attachments': bool(row['has_attachments']),
                'timestamp': row['timestamp']
            })
        
        return messages
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get all messages from a specific session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT *
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'role': row['message_role'],
                'content': row['message_content'],
                'departments': json.loads(row['departments']) if row['departments'] else [],
                'sentiment': row['sentiment'],
                'escalated': bool(row['escalated']),
                'timestamp': row['timestamp']
            })
        
        return messages
    
    def get_user_sessions(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get all sessions for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                s.*,
                COUNT(c.id) as total_messages
            FROM sessions s
            LEFT JOIN conversations c ON s.session_id = c.session_id
            WHERE s.user_id = ?
            GROUP BY s.id
            ORDER BY s.started_at DESC
            LIMIT ?
        """, (user_id, limit))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row['session_id'],
                'started_at': row['started_at'],
                'ended_at': row['ended_at'],
                'message_count': row['total_messages'],
                'departments_used': json.loads(row['departments_used']) if row['departments_used'] else []
            })
        
        return sessions
    
    # ========================================================================
    # SEARCH & ANALYTICS
    # ========================================================================
    
    def search_conversations(self, user_id: int, keyword: str, 
                            limit: int = 20) -> List[Dict]:
        """Search through user's conversation history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT *
            FROM conversations
            WHERE user_id = ? AND message_content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, f'%{keyword}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'role': row['message_role'],
                'content': row['message_content'],
                'session_id': row['session_id'],
                'timestamp': row['timestamp']
            })
        
        return results
    
    def get_department_usage_stats(self, user_id: int) -> Dict:
        """Get statistics on department usage for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT departments
            FROM conversations
            WHERE user_id = ? AND departments IS NOT NULL
        """, (user_id,))
        
        dept_counts = {}
        for row in cursor.fetchall():
            if row['departments']:
                depts = json.loads(row['departments'])
                for dept in depts:
                    dept_counts[dept] = dept_counts.get(dept, 0) + 1
        
        return dept_counts
    
    def export_user_data(self, user_id: int) -> Dict:
        """Export all user data (for GDPR compliance)"""
        user_info = self.get_user_by_id(user_id)
        conversations = self.get_conversation_history(user_id, limit=10000)
        sessions = self.get_user_sessions(user_id, limit=10000)
        stats = self.get_user_stats(user_id)
        
        return {
            'user_info': user_info,
            'conversations': conversations,
            'sessions': sessions,
            'statistics': stats,
            'exported_at': datetime.now().isoformat()
        }
    
    def get_user_by_id(self, user_id: int) -> Dict:
        """Get user information by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if user:
            return dict(user)
        return None
    
    # ========================================================================
    # MAINTENANCE
    # ========================================================================
    
    def delete_old_conversations(self, days: int = 90):
        """Delete conversations older than specified days"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM conversations
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days,))
        
        deleted = cursor.rowcount
        conn.commit()
        
        return deleted
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize database
    db = DatabaseManager("shopunow_test.db")
    
    # Create or get user
    user = db.create_or_get_user(
        mobile_number="+1234567890",
        name="John Doe",
        email="john@example.com"
    )
    
    print(f"User: {user}")
    
    # Create session
    session_id = "test_session_001"
    db.create_session(user['id'], session_id)
    
    # Save messages
    db.save_message(
        user_id=user['id'],
        session_id=session_id,
        role="user",
        content="How do I return a product?",
        departments=["customer_service"],
        sentiment="neutral"
    )
    
    db.save_message(
        user_id=user['id'],
        session_id=session_id,
        role="assistant",
        content="You can return any product within 30 days...",
        departments=["customer_service"]
    )
    
    # Get conversation history
    history = db.get_conversation_history(user['id'])
    print(f"\nConversation History: {len(history)} messages")
    
    # Get user stats
    stats = db.get_user_stats(user['id'])
    print(f"\nUser Stats: {stats}")
    
    # End session
    db.end_session(session_id, ["customer_service"])
    
    print("\n✓ Database test completed successfully!")