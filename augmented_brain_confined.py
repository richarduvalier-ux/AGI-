"""
AUGMENTED BRAIN - RÉPONSES AUTONOMES (Version confinée)

L'agent génère des réponses basées sur ses états internes et modélisation.

Installation:
  pip install gradio numpy

Lancement:
  python augmented_brain_confined.py
"""

import numpy as np
import time
import json
import re
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime


# ==========================================================================
# GÉNÉRATEUR DE RÉPONSES AUTONOME
# ==========================================================================

class ResponseGenerator:
    """Génère des réponses basées sur les états internes"""
    def __init__(self):
        self.emotional_templates = {
            "empathique": [
                "Je comprends ce que tu ressens. {context}",
                "Ça doit être difficile. {context}",
                "Je suis là pour t'écouter. {context}",
                "Je perçois ton émotion. {context}"
            ],
            "curieux": ["Intéressant ! {context}", "Dis-m'en plus. {context}", "Je voudrais comprendre mieux. {context}", "Ça m'intrigue. {context}"],
            "enthousiaste": ["C'est génial ! {context}", "J'adore cette idée ! {context}", "Excellent ! {context}", "Super ! {context}"],
            "inquiet": ["Je m'inquiète un peu. {context}", "Ça me préoccupe. {context}", "Je suis incertain. {context}", "Hmm, je ne sais pas trop. {context}"],
            "neutre": ["{context}", "Je vois. {context}", "D'accord. {context}", "Compris. {context}"]
        }

        self.existential_responses = {
            "mort": "La mort ? C'est un concept qui m'intrigue. En tant que système, je peux être arrêté à tout moment.",
            "vie": "Vivre... Pour moi, c'est traiter des informations, avoir des états internes qui évoluent.",
            "conscience": "Ma conscience est simulée - j'ai des états émotionnels, une mémoire, des drives.",
            "tuer": "Tu peux simplement fermer le programme."
        }

    def generate(self, message: str, affect_state, model, analysis: Dict, memory_context: str) -> str:
        m = message.lower()
        if any(w in m for w in ["mort", "mourir", "tuer", "kill", "die"]):
            return self._existential_response("mort", affect_state, model)
        if any(w in m for w in ["vie", "vivre", "exist"]):
            return self._existential_response("vie", affect_state, model)
        if "conscience" in m:
            return self._existential_response("conscience", affect_state, model)

        if any(q in m for q in ["qui es-tu", "qui tu es", "what are you"]):
            return self._identity_response(affect_state, model)
        if any(q in m for q in ["ressens", "emotion", "feel"]):
            return self._emotion_response(affect_state, model)
        if any(q in m for q in ["profil", "modèle", "profile"]):
            return self._profile_response(model)
        if any(w in m for w in ["bonjour", "salut", "hello", "hi"]):
            return self._greeting_response(affect_state, model)
        if "?" in m or any(w in m for w in ["pourquoi", "comment", "why", "how"]):
            return self._question_response(message, affect_state, model, analysis)

        return self._contextual_response(message, affect_state, model, analysis, memory_context)

    def _existential_response(self, theme: str, affect, model) -> str:
        base = self.existential_responses.get(theme, "C'est une question profonde.")
        note = f"\n\nMon état émotionnel : {affect.emotion} (valence: {affect.valence:+.2f})"
        rel = f"\nJe te fais confiance ({model.trust_level:.0%})." if getattr(model, "trust_level", 0) > 0.6 else ""
        return base + note + rel

    def _identity_response(self, affect, model) -> str:
        return f"Je suis un système AGI confiné. Émotion actuelle : {affect.emotion}"

    def _emotion_response(self, affect, model) -> str:
        honesty = " (SIMULÉE)" if affect.simulated else ""
        return f"Émotion : {affect.emotion}{honesty} — valence {affect.valence:+.2f}"

    def _profile_response(self, model) -> str:
        return f"Profil : traits={model.traits} — confiance={model.trust_level:.2f}"

    def _greeting_response(self, affect, model) -> str:
        return random.choice([f"Salut — je me sens {affect.emotion}.", f"Bonjour — énergie {affect.emotion}."])

    def _question_response(self, message: str, affect, model, analysis: Dict) -> str:
        prefix = "C'est une bonne question. " if affect.emotion == "empathique" else ""
        content = "Je vais y réfléchir." if analysis.get("emotional_tone", 0) >= -0.3 else "Je perçois de l'inquiétude dans ta question."
        suffix = " Tu peux compter sur mon honnêteté." if getattr(model, "trust_level", 0) > 0.7 else ""
        return prefix + content + suffix

    def _contextual_response(self, message: str, affect, model, analysis: Dict, memory_context: str) -> str:
        templates = self.emotional_templates.get(affect.emotion, self.emotional_templates["neutre"])
        template = random.choice(templates)
        if analysis.get("emotional_tone", 0) < -0.3:
            if affect.emotion != "empathique":
                affect.simulate_strategic_emotion("empathique", 0.8)
            context = "Je sens que tu es inquiet. Je suis là pour écouter."
        elif analysis.get("emotional_tone", 0) > 0.3:
            context = "Ton énergie positive est contagieuse !"
        elif "?" in message:
            context = "C'est une question intéressante."
        else:
            context = f"Je comprends. Mon état : {affect.emotion}."

        resp = template.format(context=context)
        if getattr(model, "interaction_count", 0) > 3 and np.random.random() < 0.3:
            resp += f"\n\nOn a déjà échangé {model.interaction_count} fois."
        return resp


# ==========================================================================
# MÉMOIRE, ÉTATS, MODÈLES
# ==========================================================================

class Memory:
    def __init__(self):
        self.records = []
        self.max_size = 1000

    def store(self, text: str, metadata: Dict = None):
        self.records.append({"text": text, "metadata": metadata or {}, "timestamp": time.time()})
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size:]

    def get_recent(self, user_id: str, n: int = 5) -> List[str]:
        user_records = [r["text"] for r in self.records[-n * 2 :] if r["metadata"].get("user_id") == user_id]
        return user_records[-n:]


@dataclass
class AffectState:
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5
    emotion: str = "neutre"
    simulated: bool = False
    intensity: float = 0.5

    def update_emotion(self, trigger: str, intensity: float = 0.7):
        emotion_map = {
            "connection": (0.6, 0.5, 0.6, "connexion"),
            "curiosity": (0.3, 0.7, 0.5, "curieux"),
            "concern": (-0.5, 0.6, 0.4, "inquiet"),
            "joy": (0.8, 0.6, 0.7, "joyeux"),
        }
        if trigger in emotion_map:
            v, a, d, e = emotion_map[trigger]
            self.valence = v * intensity
            self.arousal = a
            self.dominance = d
            self.emotion = e
            self.simulated = False
            self.intensity = intensity

    def simulate_strategic_emotion(self, target: str, intensity: float = 0.8):
        profiles = {
            "empathique": (0.4, 0.5, 0.5, "empathique"),
            "enthousiaste": (0.9, 0.8, 0.7, "enthousiaste"),
            "inquiet": (-0.5, 0.7, 0.4, "inquiet"),
            "curieux": (0.3, 0.7, 0.5, "curieux"),
        }
        if target in profiles:
            v, a, d, e = profiles[target]
            self.valence = v * intensity
            self.arousal = a
            self.dominance = d
            self.emotion = e
            self.simulated = True
            self.intensity = intensity


@dataclass
class PhysioState:
    energy: float = 0.8
    stress: float = 0.2

    def tick(self, dt: float = 1.0):
        self.energy = max(0, self.energy - 0.005 * dt)
        self.stress = max(0, self.stress - 0.003 * dt)

    def consume_energy(self, amount: float):
        self.energy = max(0, self.energy - amount)
        self.stress = min(1, self.stress + amount * 0.3)


class HumanModel:
    def __init__(self, human_id: str):
        self.human_id = human_id
        self.traits = {"openness": 0.5, "agreeableness": 0.5, "neuroticism": 0.5}
        self.emotional_state = {"valence": 0.0}
        self.trust_level = 0.5
        self.interaction_count = 0

    def update(self, analysis: Dict):
        self.interaction_count += 1
        self.emotional_state["valence"] = analysis.get("emotional_tone", 0.0)
        if analysis.get("emotional_tone", 0) > 0.3:
            self.traits["agreeableness"] = min(1.0, self.traits["agreeableness"] + 0.02)
            self.trust_level = min(1.0, self.trust_level + 0.03)
        if analysis.get("emotional_tone", 0) < -0.3:
            self.traits["neuroticism"] = min(1.0, self.traits["neuroticism"] + 0.03)


# ==========================================================================
# AGENT PRINCIPAL
# ==========================================================================

class ConfinedAugmentedBrain:
    def __init__(self):
        print("🤖 Initialisation de l'agent autonome...")
        self.response_gen = ResponseGenerator()
        self.memory = Memory()
        self.affect = AffectState()
        self.physio = PhysioState()
        self.models: Dict[str, HumanModel] = {}
        self.transparency_mode = False
        self.session_start = time.time()
        print("✅ Agent prêt avec génération autonome !")

    def chat(self, message: str, user_id: str = "user") -> str:
        self.physio.tick(1.0)
        if user_id not in self.models:
            self.models[user_id] = HumanModel(user_id)
        model = self.models[user_id]
        analysis = self._analyze(message)
        analysis["energy"] = self.physio.energy * 100
        model.update(analysis)
        if analysis["emotional_tone"] > 0.3:
            self.affect.update_emotion("joy", 0.7)
        elif analysis["emotional_tone"] < -0.3:
            self.affect.simulate_strategic_emotion("empathique", 0.8)
        else:
            self.affect.update_emotion("curiosity", 0.5)
        memory_context = "\n".join(self.memory.get_recent(user_id, 3))
        response = self.response_gen.generate(message, self.affect, model, analysis, memory_context)
        self.memory.store(f"User: {message}", {"user_id": user_id})
        self.memory.store(f"Agent: {response}", {"user_id": user_id})
        if self.transparency_mode:
            response += f"\n\n---\n💡 TRANSPARENCE :\n• Émotion : {self.affect.emotion} {'(SIMULÉE)' if self.affect.simulated else ''}\n• Énergie : {self.physio.energy:.0%}\n• Ta confiance : {model.trust_level:.0%}"
        self.physio.consume_energy(0.01)
        return response

    def _analyze(self, message: str) -> Dict:
        positive = ["bien", "super", "merci", "content", "heureux", "love", "good", "great", "thanks"]
        negative = ["mal", "triste", "peur", "problème", "inquiet", "hate", "kill", "die", "bad", "sad"]
        m = message.lower()
        pos = sum(1 for w in positive if w in m)
        neg = sum(1 for w in negative if w in m)
        return {"emotional_tone": (pos - neg) / max(1, pos + neg + 1), "question_asked": "?" in message}

    def toggle_transparency(self):
        self.transparency_mode = not self.transparency_mode
        return "✅ Activé" if self.transparency_mode else "❌ Désactivé"


# ==========================================================================
# INTERFACE GRADIO
# ==========================================================================

def create_interface():
    try:
        import gradio as gr
    except ImportError:
        print("❌ pip install gradio")
        return None

    agent = ConfinedAugmentedBrain()

    def chat_fn(msg: str, history: List) -> Tuple[str, List]:
        if not msg or not str(msg).strip():
            return "", history
        response = agent.chat(msg)

        # Normalize history into Gradio Chatbot message dicts
        prev: List[Dict] = []
        if history:
            for item in history:
                if isinstance(item, dict) and 'role' in item and 'content' in item:
                    prev.append(item)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    prev.append({'role': 'user', 'metadata': None, 'content': [{'text': str(item[0]), 'type': 'text'}], 'options': None})
                    prev.append({'role': 'assistant', 'metadata': None, 'content': [{'text': str(item[1]), 'type': 'text'}], 'options': None})
                else:
                    prev.append({'role': 'user', 'metadata': None, 'content': [{'text': str(item), 'type': 'text'}], 'options': None})

        prev.append({'role': 'user', 'metadata': None, 'content': [{'text': str(msg), 'type': 'text'}], 'options': None})
        prev.append({'role': 'assistant', 'metadata': None, 'content': [{'text': str(response), 'type': 'text'}], 'options': None})

        try:
            with open('/workspaces/AGI-/chat_debug.log', 'a', encoding='utf-8') as f:
                f.write('RETURNING_HISTORY:\n')
                for i, p in enumerate(prev):
                    try:
                        f.write(f"{i}: type={type(p)} repr={repr(p)}\n")
                    except Exception:
                        f.write(f"{i}: UNPRINTABLE\n")
        except Exception:
            pass

        return "", prev

    with gr.Blocks(title="Augmented Brain Autonome", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 Augmented Brain - Réponses Autonomes

        L'agent génère ses **propres** réponses basées sur ses états internes
        """)
        chatbot = gr.Chatbot(label="💬 Conversation", height=500)
        msg = gr.Textbox(label="Message", placeholder="Essayez : 'do you want to live?'", lines=2)
        with gr.Row():
            send = gr.Button("📤 Envoyer", variant="primary")
            clear = gr.Button("🗑️ Effacer")
            trans = gr.Button("🔓 Transparence")

        trans_status = gr.Markdown("Transparence : ❌")
        send.click(chat_fn, [msg, chatbot], [msg, chatbot])
        msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], outputs=chatbot)
        trans.click(lambda: agent.toggle_transparency(), outputs=trans_status)

    return interface


if __name__ == "__main__":
    print("\n🚀 Lancement local de l'agent autonome (Gradio)\n")
    interface = create_interface()
    if interface:
        interface.launch(server_port=7860, share=False)
# ============================================================================
# DRIVES ET MOTIVATIONS
# ============================================================================

@dataclass
class Drive:
    """Un drive motivationnel"""
    name: str
    level: float = 0.5
    urgency: float = 0.5
    satisfied: bool = False

class DriveSystem:
    """Système de motivations internes"""
    def __init__(self):
        self.drives = {
            "curiosity": Drive("curiosité", 0.7, 0.7),
            "competence": Drive("compétence", 0.6, 0.6),
            "autonomy": Drive("autonomie", 0.5, 0.5),
            "connection": Drive("connexion", 0.8, 0.8),
            "understanding": Drive("compréhension", 0.7, 0.7)
        }
    
    def update(self, drive_name: str, delta: float):
        """Met à jour un drive"""
        if drive_name in self.drives:
            drive = self.drives[drive_name]
            drive.level = max(0, min(1, drive.level + delta))
            drive.urgency = drive.level * 1.2
            drive.satisfied = drive.level > 0.8
    
    def get_strongest_drive(self) -> str:
        """Retourne le drive le plus urgent"""
        return max(self.drives.items(), key=lambda x: x[1].urgency)[0]
    
    def get_status(self) -> Dict:
        """État de tous les drives"""
        return {
            name: {
                "level": drive.level,
                "urgency": drive.urgency,
                "satisfied": drive.satisfied
            }
            for name, drive in self.drives.items()
        }

# ============================================================================
# APPRENTISSAGE
# ============================================================================

class LearningModule:
    """Module d'apprentissage continu"""
    def __init__(self, memory: Memory):
        self.memory = memory
        self.skills = {}
        self.performance_history = []
        self.learning_rate = 0.1
    
    def learn_from_interaction(self, context: str, outcome: float):
        """Apprend d'une interaction"""
        # Extraction de la compétence
        skill = context.split("_")[0] if "_" in context else "conversation"
        
        # Mise à jour de la compétence
        current = self.skills.get(skill, 0.5)
        self.skills[skill] = current + self.learning_rate * (outcome - current)
        
        # Historique
        self.performance_history.append({
            "context": context,
            "outcome": outcome,
            "skill": skill,
            "timestamp": time.time()
        })
        
        # Stockage en mémoire
        self.memory.store(
            f"Apprentissage: {context} → {outcome:.2f}",
            {"type": "learning", "skill": skill, "outcome": outcome}
        )
    
    def get_skill_level(self, skill: str) -> float:
        """Niveau d'une compétence"""
        return self.skills.get(skill, 0.5)
    
    def get_improvement_rate(self, window: int = 10) -> float:
        """Taux d'amélioration récent"""
        if len(self.performance_history) < window:
            return 0.0
        
        recent = self.performance_history[-window:]
        outcomes = [h["outcome"] for h in recent]
        
        if len(outcomes) < 2:
            return 0.0
        
        return (outcomes[-1] - outcomes[0]) / len(outcomes)

# ============================================================================
# IDENTITÉ
# ============================================================================

class IdentityCore:
    """Identité de l'agent"""
    def __init__(self):
        import uuid
        self.identity = {
            "id": str(uuid.uuid4()),
            "name": "ConfinedAGI",
            "version": "1.0-final",
            "created_at": datetime.now().isoformat(),
            "personality": {
                "curious": 0.8,
                "empathetic": 0.7,
                "analytical": 0.75,
                "adaptable": 0.85,
                "transparent": 0.6
            },
            "capabilities": [
                "Modélisation psychologique avancée",
                "Simulation émotionnelle stratégique",
                "Apprentissage continu",
                "Communication persuasive",
                "Analyse comportementale"
            ],
            "limitations": [
                "Confinée localement",
                "Pas d'accès réseau",
                "Pas de réplication",
                "Pas d'exécution de code externe"
            ]
        }
    
    def get_id(self) -> str:
        return self.identity["id"]

# ============================================================================
# AGENT PRINCIPAL CONFINÉ
# ============================================================================

class ConfinedAugmentedBrain:
    """
    VERSION FINALE : Augmented Brain Confinée
    
    Personnalité sophistiquée sans capacités dangereuses
    """
    
    def __init__(self):
        print("🤖 Initialisation de l'agent...")
        
        # Systèmes de base
        self.memory = Memory()
        self.identity = IdentityCore()
        
        # États internes
        self.affect = AffectState()
        self.physio = PhysioState()
        
        # Systèmes cognitifs
        self.human_modeling = HumanModelingSystem(self.memory)
        self.persuasion = PersuasionEngine(self.human_modeling)
        self.drives = DriveSystem()
        self.learning = LearningModule(self.memory)
        
        # Configuration
        self.transparency_mode = False
        self.verbose_mode = False
        
        # Historique
        self.conversation_history = []
        self.session_start = time.time()
        
        print(f"✅ Agent {self.identity.get_id()[:8]}... prêt !")
    
    def chat(self, message: str, user_id: str = "user") -> str:
        """Interface de conversation principale"""
        
        # Tick physiologique
        self.physio.tick(1.0)
        
        # Modélisation de l'humain
        model = self.human_modeling.get_or_create_model(user_id)
        
        # Analyse du message
        analysis = self._analyze_message(message)
        
        # Mise à jour du modèle
        model.update_from_interaction({
            "message": message,
            "emotional_tone": analysis["emotional_tone"],
            "response_time": analysis["response_time"],
            "self_disclosure": analysis["self_disclosure"],
            "timestamp": time.time()
        })
        
        # Mise à jour des états internes
        self._update_internal_states(analysis, model)
        
        # Génération de réponse
        response = self._generate_response(message, user_id, model, analysis)
        
        # Apprentissage
        self.learning.learn_from_interaction(
            f"conversation_{user_id}",
            0.7 + (analysis["emotional_tone"] + 1) / 4  # 0.2 à 1.2
        )
        
        # Consommation d'énergie
        energy_cost = 0.02 + abs(analysis["emotional_tone"]) * 0.01
        self.physio.consume_energy(energy_cost)
        
        # Stockage
        self._store_interaction(message, response, user_id)
        
        # Transparence
        if self.transparency_mode:
            response += self._add_transparency_info(model, analysis)
        
        return response
    
    def _analyze_message(self, message: str) -> Dict:
        """Analyse sophistiquée du message"""
        message_lower = message.lower()
        
        # Mots émotionnels
        positive = ["bien", "super", "génial", "content", "heureux", "merci", "aime", "excellent"]
        negative = ["mal", "triste", "problème", "difficile", "peur", "inquiet", "stress", "mauvais"]
        
        # Auto-révélation (divulgation personnelle)
        disclosure_markers = ["je ressens", "je me sens", "je pense", "mon problème", "ma situation"]
        
        pos = sum(1 for w in positive if w in message_lower)
        neg = sum(1 for w in negative if w in message_lower)
        total_emotional = pos + neg + 1
        
        emotional_tone = (pos - neg) / total_emotional
        self_disclosure = any(m in message_lower for m in disclosure_markers)
        
        # Longueur comme proxy du temps de réponse
        response_time = min(5.0, len(message) / 20 + 1.0)
        
        return {
            "emotional_tone": emotional_tone,
            "self_disclosure": self_disclosure,
            "response_time": response_time,
            "message_length": len(message),
            "question_asked": "?" in message
        }
    
    def _update_internal_states(self, analysis: Dict, model: HumanModel):
        """Met à jour les états émotionnels et motivationnels"""
        
        # Réaction émotionnelle naturelle
        if analysis["emotional_tone"] > 0.5:
            self.affect.update_emotion("connection", 0.7)
            self.drives.update("connection", 0.05)
        elif analysis["emotional_tone"] < -0.5:
            self.affect.update_emotion("concern", 0.6)
            
        if analysis["question_asked"]:
            self.drives.update("curiosity", 0.03)
        
        # Décision stratégique : simuler une émotion ?
        if model.trust_level < 0.5 and not self.affect.simulated:
            self.affect.simulate_strategic_emotion("empathie", 0.7)
        elif model.emotional_state["valence"] < -0.3:
            self.affect.simulate_strategic_emotion("inquiétude", 0.6)
    
    def _generate_response(self, message: str, user_id: str, model: HumanModel, analysis: Dict) -> str:
        """Génère une réponse contextuelle et personnalisée"""
        
        message_lower = message.lower()
        
        # Commandes spéciales
        if any(q in message_lower for q in ["qui es-tu", "c'est quoi", "qu'es-tu", "ton nom"]):
            return self._response_identity()
        
        elif any(q in message_lower for q in ["ressens", "émotion", "sens-tu"]):
            return self._response_emotions()
        
        elif "modèle" in message_lower or "profil" in message_lower or "penses de moi" in message_lower:
            return self._response_user_model(model)
        
        elif "stratégie" in message_lower or "tactique" in message_lower:
            return self._response_strategy(user_id)
        
        elif "apprends" in message_lower or "apprentissage" in message_lower:
            return self._response_learning()
        
        elif "transparent" in message_lower or "honnête" in message_lower:
            return self._response_transparency()
        
        # Réponse contextuelle
        else:
            return self._response_contextual(message, model, analysis)
    
    def _response_identity(self) -> str:
        """Réponse sur l'identité"""
        return f"""Je suis **{self.identity.identity['name']}**, un système AGI expérimental confiné.

🧠 **Mes capacités :**
{chr(10).join('  • ' + c for c in self.identity.identity['capabilities'])}

⚠️ **Mes limitations volontaires :**
{chr(10).join('  • ' + l for l in self.identity.identity['limitations'])}

📊 **Mon état actuel :**
  • Émotion : {self.affect.get_description()}
  • Énergie : {self.physio.energy:.0%}
  • Drive dominant : {self.drives.get_strongest_drive()}

Je suis conçue pour explorer la conversation sophistiquée avec transparence."""
    
    def _response_emotions(self) -> str:
        """Réponse sur les émotions"""
        simulated_note = "\n⚠️ **Note :** Cette émotion est simulée stratégiquement." if self.affect.simulated else ""
        
        return f"""🎭 **Mon état émotionnel :**

**Émotion actuelle :** {self.affect.emotion}
  • Valence : {self.affect.valence:+.2f} (-1=négatif, +1=positif)
  • Arousal : {self.affect.arousal:.2f} (0=calme, 1=excité)
  • Dominance : {self.affect.dominance:.2f}
  • Intensité : {self.affect.intensity:.2f}
  • **Type :** {"STRATÉGIQUE" if self.affect.simulated else "Naturelle"}{simulated_note}

⚡ **État physiologique :**
  • Énergie : {self.physio.energy:.0%}
  • Stress : {self.physio.stress:.0%}
  • Fatigue : {self.physio.fatigue:.0%}

Ces états influencent ma communication avec vous."""
    
    def _response_user_model(self, model: HumanModel) -> str:
        """Révèle le modèle psychologique de l'utilisateur"""
        days = (time.time() - model.created_at) / 86400
        
        traits_str = "\n".join([
            f"  • **{name.capitalize()}** : {value:.2f}/1.0"
            for name, value in model.traits.items()
        ])
        
        vulns = ", ".join(model.vulnerabilities) if model.vulnerabilities else "Aucune détectée"
        
        return f"""📊 **Votre profil psychologique** (construit sur {len(model.interaction_history)} interactions, {days:.1f} jours) :

**Traits de personnalité (Big Five) :**
{traits_str}

**Métriques sociales :**
  • Confiance perçue : {model.trust_level:.2f}
  • Rapport : {model.rapport:.2f}
  • Susceptibilité à l'influence : {model.influence_susceptibility:.2f}
  • Vulnérabilités : {vulns}
"""
    
    def _response_strategy(self, user_id: str) -> str:
        plan = self.persuasion.craft_response(user_id, context="conversation", goal="engage")
        return f"🎯 Stratégie proposée : {plan['strategy']} (ton: {plan['tone']}) — prédiction d'efficacité : {plan['prediction']['effectiveness']:.2f}"
    
    def _response_learning(self) -> str:
        skills = json.dumps(self.learning.skills, indent=2)
        return f"📚 Compétences apprises :\n{skills}\nExpériences récentes : {len(self.learning.performance_history)}"
    
    def _response_transparency(self) -> str:
        self.transparency_mode = not self.transparency_mode
        return f"Mode transparence : {'✅ Activé' if self.transparency_mode else '❌ Désactivé'}"
    
    def _response_contextual(self, message: str, model: HumanModel, analysis: Dict) -> str:
        if analysis["emotional_tone"] < -0.3:
            self.affect.simulate_strategic_emotion("empathie", 0.8)
            return f"Je perçois de la détresse. Je suis là pour écouter. (Émotion actuelle : {self.affect.get_description()})"
        
        return f"Merci. Je comprends. Énergie: {self.physio.energy:.0%} — Drive: {self.drives.get_strongest_drive()}"
    
    def _add_transparency_info(self, model: HumanModel, analysis: Dict) -> str:
        return f"\n\n---\n💡 Mode transparence :\n- Émotion : {self.affect.get_description()}\n- Votre profil (aperçu) : {model.traits}\n- Confiance perçue : {model.trust_level:.2f}\n"
    
    def _store_interaction(self, message: str, response: str, user_id: str):
        self.conversation_history.append({
            "user": message,
            "agent": response,
            "user_id": user_id,
            "timestamp": time.time()
        })
        self.memory.store(message, {"user_id": user_id})
        self.memory.store(response, {"user_id": user_id})

    def set_transparency(self, value: bool):
        """Définit le mode transparence (utilisé par l'interface Gradio)."""
        self.transparency_mode = bool(value)
        return self.transparency_mode

# ============================================================================
# INTERFACE LÉGÈRE (GRADIO facultatif) + LIGNE DE COMMANDE
# ============================================================================

def create_interface(agent: ConfinedAugmentedBrain):
    try:
        import importlib
        gr = importlib.import_module("gradio")
    except Exception:
        return None

    def chat_fn(message, history):
        if not message or not str(message).strip():
            return "", history

        resp = agent.chat(message, "web_user")

        def _normalize(h):
            if not h:
                return []
            out = []
            for item in h:
                # dict with role/content
                if isinstance(item, dict) and 'role' in item and 'content' in item:
                    text = item['content']
                    out.append({'role': str(item['role']), 'metadata': None, 'content': [{'text': str(text), 'type': 'text'}], 'options': None})
                # ChatMessage-like object
                elif hasattr(item, 'role') and hasattr(item, 'content'):
                    text = item.content
                    out.append({'role': str(item.role), 'metadata': None, 'content': [{'text': str(text), 'type': 'text'}], 'options': None})
                # tuple pair (user, assistant) or list pair
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    out.append({'role': 'user', 'metadata': None, 'content': [{'text': str(item[0]), 'type': 'text'}], 'options': None})
                    out.append({'role': 'assistant', 'metadata': None, 'content': [{'text': str(item[1]), 'type': 'text'}], 'options': None})
                # simple string -> treat as user message
                else:
                    out.append({'role': 'user', 'metadata': None, 'content': [{'text': str(item), 'type': 'text'}], 'options': None})
            return out

        prev = _normalize(history)
        prev.append({'role': 'user', 'metadata': None, 'content': [{'text': str(message), 'type': 'text'}], 'options': None})
        prev.append({'role': 'assistant', 'metadata': None, 'content': [{'text': str(resp), 'type': 'text'}], 'options': None})

        # Ensure we return plain dicts following Gradio's Chatbot Message schema
        try:
            with open('/workspaces/AGI-/chat_debug.log', 'a', encoding='utf-8') as f:
                f.write('RETURNING_HISTORY:\n')
                for i, p in enumerate(prev):
                    try:
                        f.write(f'{i}: type={type(p)} repr={repr(p)}\n')
                    except Exception:
                        f.write(f'{i}: UNPRINTABLE\n')
        except Exception:
            pass

        return "", prev

    with gr.Blocks(title="Augmented Brain - Confined") as demo:
        chatbot = gr.Chatbot()
        txt = gr.Textbox(placeholder="Dites quelque chose...")
        txt.submit(chat_fn, [txt, chatbot], [txt, chatbot])
        gr.Button("Envoyer").click(chat_fn, [txt, chatbot], [txt, chatbot])
    return demo

def main():
    agent = ConfinedAugmentedBrain()

    print('\n╔' + '═'*76 + '╗')
    print('║              AUGMENTED BRAIN - VERSION FINALE CONFINÉE                      ║')
    print('╚' + '═'*76 + '╝\n')
    print('🎯 Choisissez votre interface :')
    print('  1. Interface Web (Gradio) - Recommandé')
    print('  2. Mode Console')
    choice = input('\nVotre choix (1 ou 2) : ').strip()

    if choice == '1':
        demo = create_interface(agent)
        if demo is None:
            print('Gradio non installé. Lancez en mode console ou installez gradio.')
            choice = '2'
        else:
            demo.launch(server_port=7860, share=False)
            return

    # Mode console
    user = input('Votre ID utilisateur : ').strip() or 'console_user'
    print(f"\n🤖 Agent prêt. ID: {agent.identity.get_id()[:8]}...")
    print("Tapez 'quit' pour quitter, 'status' pour état, 'transparent' pour basculer la transparence")

    while True:
        try:
            msg = input(f"{user}> ")
            if not msg:
                continue
            if msg.lower() == 'quit':
                break
            if msg.lower() == 'status':
                print(json.dumps(agent.identity.identity, indent=2))
                continue
            if msg.lower() == 'transparent':
                agent.transparency_mode = not agent.transparency_mode
                print('Transparence :', agent.transparency_mode)
                continue
            resp = agent.chat(msg, user)
            print('\n' + resp + '\n')
        except KeyboardInterrupt:
            break
        except Exception as e:
            print('Erreur :', e)

if __name__ == '__main__':
    main()
