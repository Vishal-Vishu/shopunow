"""
Department Persona Showcase
============================
Demonstrates the unique personas and communication styles
of each department agent with side-by-side comparisons.
"""

import os
from department_agents import DepartmentAgentManager, DEPARTMENT_AGENT_CONFIGS
from typing import Dict, List

class PersonaShowcase:
    """Showcase and test department personas"""
    
    def __init__(self, api_key: str):
        self.manager = DepartmentAgentManager(api_key)
    
    def display_all_personas(self):
        """Display all persona information"""
        print("\n" + "=" * 100)
        print("🎭 SHOPUNOW DEPARTMENT PERSONAS - COMPLETE OVERVIEW")
        print("=" * 100 + "\n")
        
        personas = self.manager.list_all_personas()
        
        for dept_id, info in personas.items():
            config = DEPARTMENT_AGENT_CONFIGS[dept_id]
            
            print(f"{'='*100}")
            print(f"🏢 DEPARTMENT: {dept_id.upper().replace('_', ' ')}")
            print(f"{'='*100}")
            print(f"\n👤 Persona: {info['name']}")
            print(f"💼 Role: {info['role']}")
            print(f"🎨 Personality: {info['personality']}")
            print(f"💬 Communication Style: {config['communication_style']}")
            print(f"🎯 Tone: {info['tone']}")
            print(f"📏 Formality Level: {config['formality_level']}")
            print(f"❤️  Empathy Level: {config['empathy_level']}")
            print(f"\n📚 Core Expertise:")
            for expertise in info['expertise']:
                print(f"   • {expertise}")
            print(f"\n🎪 Response Pattern:")
            print(f"   Length: {config['response_length']}")
            print(f"   Proactive: {'Yes' if config['proactive_suggestions'] else 'No'}")
            print(f"\n{'='*100}\n")
    
    def compare_persona_responses(self, query: str, departments: List[str] = None):
        """Compare how different personas respond to the same query"""
        if departments is None:
            departments = ["customer_service", "sales", "technical_support"]
        
        print("\n" + "=" * 100)
        print(f"🔍 PERSONA COMPARISON - SAME QUERY, DIFFERENT STYLES")
        print("=" * 100)
        print(f"\n📝 Query: \"{query}\"\n")
        print("=" * 100 + "\n")
        
        for dept_id in departments:
            agent = self.manager.get_agent(dept_id)
            if not agent:
                continue
            
            persona_info = agent.get_persona_info()
            
            print(f"{'─'*100}")
            print(f"👤 {persona_info['name']} ({persona_info['role']})")
            print(f"🎯 Personality: {persona_info['personality']}")
            print(f"{'─'*100}\n")
            
            response = agent.generate_response(query)
            print(response)
            print(f"\n{'─'*100}\n")
    
    def test_persona_characteristics(self):
        """Test specific persona characteristics with targeted queries"""
        test_scenarios = [
            {
                "scenario": "Frustrated Customer",
                "query": "This is ridiculous! My order is 2 weeks late and nobody is helping me!",
                "departments": ["customer_service"],
                "expected": "Empathetic, solution-focused, calming"
            },
            {
                "scenario": "Technical Question",
                "query": "My laptop screen is flickering and has lines across it.",
                "departments": ["technical_support"],
                "expected": "Methodical troubleshooting, step-by-step"
            },
            {
                "scenario": "Product Inquiry",
                "query": "I'm looking for a good laptop for video editing. What do you recommend?",
                "departments": ["product_information"],
                "expected": "Enthusiastic, detailed specs, recommendations"
            },
            {
                "scenario": "Deal Seeker",
                "query": "Do you have any sales on electronics right now?",
                "departments": ["sales"],
                "expected": "Energetic, promotional, value-focused"
            },
            {
                "scenario": "HR Policy Question",
                "query": "How many vacation days do I have?",
                "departments": ["hr"],
                "expected": "Professional, policy-based, supportive"
            },
            {
                "scenario": "IT Access Issue",
                "query": "I can't access my email. Can you help?",
                "departments": ["it_support"],
                "expected": "Security-conscious, step-by-step, technical"
            },
            {
                "scenario": "Expense Question",
                "query": "Can I expense this $500 hotel bill from my business trip?",
                "departments": ["finance"],
                "expected": "Precise, policy-focused, compliance-oriented"
            },
            {
                "scenario": "Warehouse Request",
                "query": "I need to request inventory for my department.",
                "departments": ["operations"],
                "expected": "Procedural, timeline-focused, practical"
            }
        ]
        
        print("\n" + "=" * 100)
        print("🧪 PERSONA CHARACTERISTICS TEST - Evaluating Unique Behaviors")
        print("=" * 100 + "\n")
        
        for scenario in test_scenarios:
            print(f"\n{'='*100}")
            print(f"📋 Scenario: {scenario['scenario']}")
            print(f"{'='*100}")
            print(f"\n💬 Query: \"{scenario['query']}\"")
            print(f"🎯 Expected Characteristics: {scenario['expected']}\n")
            print(f"{'─'*100}\n")
            
            for dept_id in scenario['departments']:
                agent = self.manager.get_agent(dept_id)
                if not agent:
                    continue
                
                persona_info = agent.get_persona_info()
                print(f"👤 Response from {persona_info['name']}:\n")
                
                response = agent.generate_response(scenario['query'])
                print(response)
                print(f"\n{'─'*100}\n")
    
    def show_persona_evolution(self, query: str, context: str):
        """Show how a persona responds with additional context"""
        dept_id = "customer_service"
        agent = self.manager.get_agent(dept_id)
        persona_info = agent.get_persona_info()
        
        print("\n" + "=" * 100)
        print(f"📊 PERSONA EVOLUTION - How {persona_info['name']} Adapts to Context")
        print("=" * 100 + "\n")
        
        print(f"{'─'*100}")
        print("Response WITHOUT Context:")
        print(f"{'─'*100}\n")
        response1 = agent.generate_response(query)
        print(response1)
        
        print(f"\n{'─'*100}")
        print("Response WITH Additional Context:")
        print(f"{'─'*100}\n")
        print(f"Context: {context}\n")
        response2 = agent.generate_response(query, context=context)
        print(response2)
        print(f"\n{'─'*100}\n")
    
    def generate_persona_summary(self):
        """Generate a markdown summary of all personas"""
        personas = self.manager.list_all_personas()
        
        summary = "# ShopUNow Department Personas\n\n"
        summary += "## Overview\n\n"
        summary += "Each department has a dedicated AI agent with a unique persona, expertise, and communication style.\n\n"
        
        for dept_id, info in personas.items():
            config = DEPARTMENT_AGENT_CONFIGS[dept_id]
            
            summary += f"## {dept_id.upper().replace('_', ' ')}\n\n"
            summary += f"**Persona:** {info['name']}\n\n"
            summary += f"**Role:** {info['role']}\n\n"
            summary += f"**Personality:** {info['personality']}\n\n"
            summary += f"**Communication Style:** {config['communication_style']}\n\n"
            summary += f"**Tone:** {info['tone']}\n\n"
            summary += f"**Expertise:**\n"
            for exp in info['expertise']:
                summary += f"- {exp}\n"
            summary += "\n"
            summary += f"**Key Characteristics:**\n"
            summary += f"- Empathy Level: {config['empathy_level']}\n"
            summary += f"- Formality Level: {config['formality_level']}\n"
            summary += f"- Response Length: {config['response_length']}\n"
            summary += f"- Proactive Suggestions: {'Yes' if config['proactive_suggestions'] else 'No'}\n\n"
            summary += "---\n\n"
        
        return summary

# ============================================================================
# MAIN - RUN SHOWCASE
# ============================================================================

def main():
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    if api_key == "your-api-key-here":
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    showcase = PersonaShowcase(api_key)
    
    # Menu
    print("\n" + "=" * 100)
    print("🎭 ShopUNow Department Persona Showcase")
    print("=" * 100 + "\n")
    print("Choose an option:")
    print("1. Display All Personas Overview")
    print("2. Compare Persona Responses (Same Query)")
    print("3. Test Persona Characteristics")
    print("4. Show Persona Evolution (With Context)")
    print("5. Generate Persona Summary (Markdown)")
    print("6. Run All Demos")
    print("\n")
    
    choice = input("Enter choice (1-6): ").strip()
    
    if choice == "1":
        showcase.display_all_personas()
    
    elif choice == "2":
        query = "I need help with my order"
        showcase.compare_persona_responses(query)
    
    elif choice == "3":
        showcase.test_persona_characteristics()
    
    elif choice == "4":
        query = "How do I return this item?"
        context = "Customer purchased a laptop 3 weeks ago for $1,200. Item arrived damaged."
        showcase.show_persona_evolution(query, context)
    
    elif choice == "5":
        summary = showcase.generate_persona_summary()
        print(summary)
        
        # Save to file
        with open("persona_summary.md", "w") as f:
            f.write(summary)
        print("\n✅ Summary saved to persona_summary.md")
    
    elif choice == "6":
        print("\n🚀 Running Complete Showcase...\n")
        showcase.display_all_personas()
        input("\nPress Enter to continue to comparisons...")
        showcase.compare_persona_responses("I need help with my order")
        input("\nPress Enter to continue to characteristics test...")
        showcase.test_persona_characteristics()
        print("\n✅ Complete showcase finished!")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
