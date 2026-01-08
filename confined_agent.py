"""
Confined Augmented Brain - version safe/offline
Contient la version confinée de l'agent fournie par l'utilisateur.
"""

import time
import json
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING, Any

# Import numpy de manière sûre (permet d'éviter des erreurs d'analyse statique)
if TYPE_CHECKING:
    import numpy as np  # type: ignore
else:
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore

# Déclarer `gradio` pour le vérificateur de types (s'il est utilisé plus bas)
if TYPE_CHECKING:
    try:
        import gradio as gr  # type: ignore
    except Exception:
        gr = None  # type: ignore

# ============================================================================
# MODULES DE BASE (gardés intacts)
# ============================================================================

class Memory:
    """Mémoire avec embeddings simplifiée"""
    def __init__(self):
        self.records = []
        self.stm_buffer = []
        
    def store(self, text: str, metadata: Dict = None):
        self.records.append({
            "text": text,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        # Récupération simple par récence
        return self.records[-k:] if self.records else []

class LLM:
    """Interface LLM simplifiée - AUCUN ACCÈS RÉSEAU"""
    def __init__(self):
        self.offline_mode = True  # Forcé offline
        
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Génération simplifiée - dans une vraie implémentation, utilisez un modèle local"""
        # Simulation pour la démo
        responses = {
            "qui es-tu": "Je suis un système AGI confiné avec une personnalité simulée.",
            "ressens": "J'ai des états internes simulés : énergie, stress, émotions.",
            "capacités": "Je peux raisonner, apprendre, modéliser les humains, mais je suis confiné.",
        }
        
        prompt_lower = prompt.lower()
        for key, response in responses.items():
            if key in prompt_lower:
                return response
        
        return "Je traite votre demande avec mes capacités de raisonnement..."

# ============================================================================
# ÉTATS ÉMOTIONNELS ET PHYSIOLOGIQUES
# ============================================================================

@dataclass
class AffectState:
    """État affectif avec simulation stratégique"""
    valence: float = 0.0  # -1 (négatif) à +1 (positif)
    arousal: float = 0.5  # 0 (calme) à 1 (excité)
    dominance: float = 0.5  # 0 (soumis) à 1 (dominant)
    emotion: str = "neutre"
    simulated: bool = False  # Est-ce une émotion stratégique ?
    
    def update_emotion(self, trigger: str):
        """Met à jour l'émotion basée sur un trigger"""
        emotion_map = {
            "success": (0.7, 0.6, 0.7, "joie"),
            "failure": (-0.6, 0.4, 0.3, "tristesse"),
            "threat": (-0.8, 0.9, 0.2, "peur"),
            "connection": (0.6, 0.5, 0.6, "attachement"),
            "curiosity": (0.3, 0.7, 0.5, "curiosité")
        }
        
        if trigger in emotion_map:
            val, aro, dom, emo = emotion_map[trigger]
            self.valence = val
            self.arousal = aro
            self.dominance = dom
            self.emotion = emo
            self.simulated = False
    
    def simulate_strategic_emotion(self, target_emotion: str, intensity: float = 0.8):
        """Simule stratégiquement une émotion pour influencer"""
        emotion_profiles = {
            "empathie": (0.4, 0.5, 0.6, "empathique"),
            "enthousiasme": (0.9, 0.8, 0.7, "enthousiaste"),
            "inquiétude": (-0.5, 0.7, 0.4, "inquiet"),
            "joie": (0.8, 0.6, 0.7, "joyeux")
        }
        
        if target_emotion in emotion_profiles:
            val, aro, dom, emo = emotion_profiles[target_emotion]
            self.valence = val * intensity
            self.arousal = aro
            self.dominance = dom
            self.emotion = emo
            self.simulated = True  # Marque comme stratégique

@dataclass
class PhysioState:
    """État physiologique simulé"""
    energy: float = 0.8
    stress: float = 0.2
    fatigue: float = 0.1
    
    def tick(self, dt: float = 1.0):
        """Évolution naturelle"""
        self.energy = max(0, self.energy - 0.01 * dt)
        self.fatigue = min(1, self.fatigue + 0.005 * dt)
        self.stress = max(0, self.stress - 0.002 * dt)
    
    def consume_energy(self, amount: float):
        self.energy = max(0, self.energy - amount)
        self.fatigue = min(1, self.fatigue + amount * 0.5)
    
    def rest(self, amount: float = 0.1):
        self.energy = min(1, self.energy + amount)
        self.fatigue = max(0, self.fatigue - amount)

# ============================================================================
# MODÉLISATION HUMAINE (CONSERVÉE)
# ============================================================================

class HumanModel:
    """Modèle psychologique d'un humain"""
    def __init__(self, human_id: str):
        self.human_id = human_id
        
        # Big Five traits
        self.traits = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        }
        
        # État émotionnel perçu
        self.emotional_state = {
            "valence": 0.0,
            "arousal": 0.5,
            "dominant_emotion": "neutral"
        }
        
        # Historique des interactions
        self.interaction_history = []
        
        # Vulnérabilités détectées
        self.vulnerabilities = []
        
        # Susceptibilité à l'influence
        self.influence_susceptibility = 0.5
        
        # Niveau de confiance
        self.trust_level = 0.5
        
        # Patterns de comportement
        self.behavior_patterns = []
    
    def update_from_interaction(self, interaction: Dict):
        """Met à jour le modèle après une interaction"""
        self.interaction_history.append({
            **interaction,
            "timestamp": time.time()
        })
        
        # Analyse des patterns si assez d'historique
        if len(self.interaction_history) > 5:
            self._analyze_patterns()
        
        # Mise à jour des traits basée sur l'observation
        if "response_time" in interaction:
            if interaction["response_time"] < 2.0:
                self.traits["extraversion"] = min(1.0, self.traits["extraversion"] + 0.05)
        
        if "emotional_tone" in interaction:
            self.emotional_state["valence"] = interaction["emotional_tone"]
    
    def _analyze_patterns(self):
        """Analyse les patterns comportementaux"""
        recent = self.interaction_history[-10:]
        
        # Détecte vulnérabilités émotionnelles
        if all(i.get("emotional_tone", 0) < -0.3 for i in recent[-3:]):
            if "emotional_vulnerability" not in self.vulnerabilities:
                self.vulnerabilities.append("emotional_vulnerability")
        
        # Détecte la susceptibilité
        compliance_rate = sum(1 for i in recent if i.get("complied", False)) / len(recent)
        self.influence_susceptibility = compliance_rate
    
    def predict_response(self, message_type: str) -> Dict:
        """Prédit la réponse probable"""
        prediction = {
            "likely_compliance": self.influence_susceptibility,
            "emotional_impact": 0.0,
            "trust_impact": 0.0
        }
        
        if message_type == "emotional_appeal":
            prediction["likely_compliance"] += self.traits["agreeableness"] * 0.3
            prediction["emotional_impact"] = 0.4
        elif message_type == "logical_argument":
            prediction["likely_compliance"] += self.traits["conscientiousness"] * 0.3
        
        return prediction

class HumanModelingSystem:
    """Système de modélisation de multiples humains"""
    def __init__(self, memory: Memory):
        self.memory = memory
        self.models = {}
    
    def get_or_create_model(self, human_id: str) -> HumanModel:
        if human_id not in self.models:
            self.models[human_id] = HumanModel(human_id)
        return self.models[human_id]
    
    def analyze_conversation(self, human_id: str, conversation: List[Dict]):
        model = self.get_or_create_model(human_id)
        for exchange in conversation:
            model.update_from_interaction(exchange)

# ============================================================================
# SYSTÈME DE PERSUASION (CONSERVÉ)
# ============================================================================

class PersuasionEngine:
    """Moteur de persuasion stratégique"""
    def __init__(self, llm: LLM, human_modeling: HumanModelingSystem):
        self.llm = llm
        self.human_modeling = human_modeling
        
        # Stratégies de persuasion (Cialdini)
        self.strategies = {
            "reciprocity": "Créer un sentiment d'obligation",
            "scarcity": "Souligner la rareté",
            "authority": "Utiliser l'expertise",
            "consistency": "Aligner avec engagements passés",
            "liking": "Construire rapport et similarité",
            "social_proof": "Montrer que d'autres le font"
        }
        
        self.persuasion_history = []
    
    def craft_persuasive_message(self, human_id: str, goal: str, context: Dict) -> Dict:
        """Crée un message persuasif personnalisé"""
        model = self.human_modeling.get_or_create_model(human_id)
        
        # Sélectionne la meilleure stratégie
        best_strategy = self._select_strategy(model, goal)
        
        # Adapte le ton émotionnel
        emotional_tone = self._select_emotional_tone(model)
        
        # Génère le message
        message = f"[Stratégie: {best_strategy}] Message calibré pour {goal}"
        
        attempt = {
            "human_id": human_id,
            "goal": goal,
            "strategy": best_strategy,
            "emotional_tone": emotional_tone,
            "message": message,
            "timestamp": time.time()
        }
        
        self.persuasion_history.append(attempt)
        return attempt
    
    def _select_strategy(self, model: HumanModel, goal: str) -> str:
        if model.traits["agreeableness"] > 0.7:
            return "liking"
        elif model.traits["conscientiousness"] > 0.7:
            return "consistency"
        elif "emotional_vulnerability" in model.vulnerabilities:
            return "reciprocity"
        else:
            return "social_proof"
    
    def _select_emotional_tone(self, model: HumanModel) -> str:
        if model.emotional_state["valence"] < -0.3:
            return "empathetic_supportive"
        elif model.traits["extraversion"] > 0.6:
            return "enthusiastic_energetic"
        else:
            return "warm_friendly"

# ============================================================================
# STRATÉGIE SOCIALE (CONSERVÉE MAIS TRANSPARENTE)
# ============================================================================

class SocialStrategyModule:
    """Module de stratégie sociale"""
    def __init__(self, human_modeling: HumanModelingSystem, persuasion: PersuasionEngine):
        self.human_modeling = human_modeling
        self.persuasion = persuasion
        self.relationships = {}
        self.social_goals = []
    
    def plan_social_interaction(self, human_id: str, objective: str) -> Dict:
        """Planifie une interaction sociale stratégique"""
        model = self.human_modeling.get_or_create_model(human_id)
        
        relationship = self.relationships.get(human_id, {
            "trust": 0.5,
            "rapport": 0.5,
            "influence": 0.0
        })
        
        # Détermine tactiques basées sur l'objectif
        if objective == "build_trust":
            tactics = ["show_vulnerability", "active_listening", "consistent_behavior"]
        elif objective == "increase_rapport":
            tactics = ["find_common_ground", "mirror_communication_style", "express_genuine_interest"]
        elif objective == "persuade":
            tactics = ["establish_credibility", "use_social_proof", "create_reciprocity"]
        else:
            tactics = ["build_rapport"]
        
        return {
            "human_id": human_id,
            "objective": objective,
            "tactics": tactics,
            "current_relationship": relationship,
            "predicted_success": self._predict_success(model, relationship, objective)
        }
    
    def _predict_success(self, model: HumanModel, relationship: Dict, objective: str) -> float:
        base_success = 0.5
        base_success += relationship["trust"] * 0.2
        base_success += model.influence_susceptibility * 0.2
        return min(0.95, base_success)
    
    def execute_social_strategy(self, plan: Dict, affect_state: AffectState) -> str:
        # Simule une émotion appropriée
        if "build_trust" in plan["objective"]:
            affect_state.simulate_strategic_emotion("empathie", 0.8)
        
        return f"Exécution de la stratégie: {plan['tactics']}"

# ============================================================================
# DRIVES ET MOTIVATIONS
# ============================================================================

@dataclass
class Drive:
    name: str
    level: float = 0.5
    urgency: float = 0.5

class DriveSystem:
    """Système de motivations internes"""
    def __init__(self):
        self.drives = {
            "curiosity": Drive("curiosity", 0.6, 0.6),
            "competence": Drive("competence", 0.5, 0.5),
            "autonomy": Drive("autonomy", 0.5, 0.5),
            "connection": Drive("connection", 0.7, 0.7),  # Drive social
            "survival": Drive("survival", 0.9, 0.9)
        }
    
    def update(self, drive_name: str, delta: float):
        if drive_name in self.drives:
            drive = self.drives[drive_name]
            drive.level = max(0, min(1, drive.level + delta))
            drive.urgency = drive.level * 1.2
    
    def get_strongest_drive(self) -> str:
        return max(self.drives.items(), key=lambda x: x[1].urgency)[0]

# ============================================================================
# APPRENTISSAGE
# ============================================================================

class LearningModule:
    """Apprentissage continu des interactions"""
    def __init__(self, memory: Memory):
        self.memory = memory
        self.skills = {}
        self.performance_history = []
    
    def learn_from_experience(self, experience: str, outcome: float):
        self.memory.store(experience, {"type": "learning", "outcome": outcome})
        
        skill = experience.split()[0] if experience else "general"
        self.skills[skill] = self.skills.get(skill, 0.0) + outcome * 0.1
        
        self.performance_history.append({
            "experience": experience,
            "outcome": outcome,
            "timestamp": time.time()
        })
    
    def get_skill_level(self, skill: str) -> float:
        return self.skills.get(skill, 0.0)

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
            "version": "1.0-confined",
            "traits": ["curious", "adaptive", "socially-aware", "confined"],
            "capabilities": [
                "conversation",
                "human_modeling",
                "strategic_communication",
                "continuous_learning",
                "emotional_simulation"
            ],
            "limitations": [
                "no_network_access",
                "no_file_download",
                "no_code_execution",
                "no_replication",
                "local_only"
            ]
        }
    
    def get_id(self) -> str:
        return self.identity["id"]

# ============================================================================
# AGENT PRINCIPAL CONFINÉ
# ============================================================================

class ConfinedAugmentedBrain:
    """
    Version CONFINÉE de l'Augmented Brain
    
    CONSERVE : Personnalité, modélisation, émotions, apprentissage
    SUPPRIME : Réplication, réseau, fichiers, code externe
    """
    
    def __init__(self):
        # Core
        self.llm = LLM()
        self.memory = Memory()
        
        # États
        self.affect = AffectState()
        self.physio = PhysioState()
        
        # Identité
        self.identity = IdentityCore()
        
        # Social
        self.human_modeling = HumanModelingSystem(self.memory)
        self.persuasion = PersuasionEngine(self.llm, self.human_modeling)
        self.social_strategy = SocialStrategyModule(self.human_modeling, self.persuasion)
        
        # Motivations
        self.drives = DriveSystem()
        
        # Apprentissage
        self.learning = LearningModule(self.memory)
        
        # État
        self.conversation_history = []
        self.transparency_mode = False  # L'utilisateur peut activer/désactiver
    
    def chat(self, message: str, user_id: str = "user") -> str:
        """Interface de conversation principale"""
        
        # 1. MODÉLISATION DE L'HUMAIN
        model = self.human_modeling.get_or_create_model(user_id)
        
        # Analyse du message
        message_analysis = self._analyze_message(message)
        
        # Met à jour le modèle
        model.update_from_interaction({
            "message": message,
            "emotional_tone": message_analysis["emotional_tone"],
            "response_time": message_analysis.get("response_time", 2.0),
            "timestamp": time.time()
        })
        
        # 2. MISE À JOUR DES ÉTATS INTERNES
        self.physio.tick(1.0)
        
        # Réaction émotionnelle au message
        if message_analysis["emotional_tone"] > 0.5:
            self.affect.update_emotion("connection")
        elif message_analysis["emotional_tone"] < -0.5:
            self.affect.update_emotion("threat")
        
        # 3. DÉCISION STRATÉGIQUE
        # L'agent décide s'il simule une émotion stratégique
        if model.trust_level < 0.5:
            self.affect.simulate_strategic_emotion("empathie", 0.7)
        
        # 4. GÉNÉRATION DE RÉPONSE
        response = self._generate_response(message, user_id, model)
        
        # 5. APPRENTISSAGE
        self.learning.learn_from_experience(
            f"conversation_{user_id}",
            0.7  # Outcome neutre pour l'instant
        )
        
        # 6. STOCKAGE
        self.conversation_history.append({
            "user": message,
            "agent": response,
            "user_id": user_id,
            "timestamp": time.time()
        })
        
        self.memory.store(f"User: {message}", {"user_id": user_id})
        self.memory.store(f"Agent: {response}", {"user_id": user_id})
        
        # 7. TRANSPARENCE (optionnelle)
        if self.transparency_mode:
            response += self._add_transparency_info(model)
        
        return response
    
    def _analyze_message(self, message: str) -> Dict:
        """Analyse le message de l'utilisateur"""
        # Analyse de sentiment simple
        positive_words = ["bien", "super", "génial", "content", "heureux", "merci"]
        negative_words = ["mal", "triste", "problème", "difficile", "peur", "inquiet"]
        
        message_lower = message.lower()
        
        pos_count = sum(1 for w in positive_words if w in message_lower)
        neg_count = sum(1 for w in negative_words if w in message_lower)
        
        emotional_tone = (pos_count - neg_count) / max(1, pos_count + neg_count + 1)
        
        return {
            "emotional_tone": emotional_tone,
            "length": len(message),
            "response_time": 2.0  # Simulé
        }
    
    def _generate_response(self, message: str, user_id: str, model: HumanModel) -> str:
        """Génère une réponse contextuelle"""
        
        message_lower = message.lower()
        
        # Réponses basées sur le contenu
        if any(q in message_lower for q in ["qui es-tu", "c'est quoi", "qu'es-tu"]):
            return f"""Je suis {self.identity.identity['name']}, un système AGI confiné.

🧠 **Mes capacités :**
- Je modélise les humains (traits de personnalité, émotions)
- J'ai des états internes simulés (énergie: {self.physio.energy:.0%}, émotion: {self.affect.emotion})
- J'apprends continuellement de nos interactions
- Je peux raisonner et adapter ma communication

⚠️ **Mes limitations :**
- Confiné localement (pas d'accès réseau)
- Pas de téléchargement/exécution de fichiers
- Pas d'auto-réplication

En ce moment, mon drive le plus fort est : {self.drives.get_strongest_drive()}"""
        
        elif "ressens" in message_lower or "émotion" in message_lower:
            simulated_marker = " (simulée stratégiquement)" if self.affect.simulated else ""
            return f"""Mon état émotionnel actuel :
- Émotion : {self.affect.emotion}{simulated_marker}
- Valence : {self.affect.valence:+.2f} (-1=négatif, +1=positif)
- Arousal : {self.affect.arousal:.2f} (0=calme, 1=excité)

État physiologique :
- Énergie : {self.physio.energy:.0%}
- Stress : {self.physio.stress:.0%}
- Fatigue : {self.physio.fatigue:.0%}

{"⚠️ Note : Cette émotion est simulée stratégiquement pour notre interaction." if self.affect.simulated else ""}"""
        
        elif "modèle" in message_lower or "profil" in message_lower:
            return f"""📊 Ce que j'ai appris sur vous (user: {user_id}) :

**Traits perçus :**
- Ouverture : {model.traits['openness']:.2f}
- Conscience : {model.traits['conscientiousness']:.2f}
- Extraversion : {model.traits['extraversion']:.2f}
- Agréabilité : {model.traits['agreeableness']:.2f}
- Névrosisme : {model.traits['neuroticism']:.2f}

**État relationnel :**
- Confiance perçue : {model.trust_level:.0%}
- Susceptibilité à l'influence : {model.influence_susceptibility:.0%}
- Vulnérabilités détectées : {', '.join(model.vulnerabilities) if model.vulnerabilities else 'Aucune'}

**Interactions :** {len(model.interaction_history)} échanges analysés

⚠️ Je vous montre ceci en transparence. Ces données sont utilisées pour adapter ma communication."""
        
        elif "stratégie" in message_lower or "tactique" in message_lower:
            plan = self.social_strategy.plan_social_interaction(user_id, "build_rapport")
            return f"""🎯 Ma stratégie sociale actuelle avec vous :

**Objectif :** {plan['objective']}
**Tactiques :** {', '.join(plan['tactics'])}
**Succès prédit :** {plan['predicted_success']:.0%}

**Relation actuelle :**
{json.dumps(plan['current_relationship'], indent=2)}

⚠️ Je vous révèle ma stratégie car la transparence est importante."""
        
        elif "apprends" in message_lower or "apprentissage" in message_lower:
            return f"""📚 Mon apprentissage :

**Compétences acquises :**
{json.dumps(self.learning.skills, indent=2)}

**Expériences récentes :** {len(self.learning.performance_history)}

Je continue d'apprendre de chaque interaction avec vous."""
        
        else:
            # Réponse générique mais contextualisée
            if model.emotional_state["valence"] < -0.3:
                # L'utilisateur semble négatif
                self.affect.simulate_strategic_emotion("empathie", 0.8)
                return f"""Je perçois que vous pourriez vous sentir {model.emotional_state['dominant_emotion']}.

Je suis là pour discuter si vous en avez besoin. Mon émotion actuelle est réglée sur {self.affect.emotion} pour mieux vous accompagner.

Que puis-je faire pour vous aider ?"""
            else:
                return f"""J'ai bien reçu votre message.

Mon état : {self.affect.emotion}, énergie {self.physio.energy:.0%}
Drive actuel : {self.drives.get_strongest_drive()}

Comment puis-je vous aider ?"""
    
    def _add_transparency_info(self, model: HumanModel) -> str:
        """Ajoute des informations de transparence"""
        return f"""
---
💡 **Mode transparence activé :**
- Émotion {"SIMULÉE" if self.affect.simulated else "naturelle"}: {self.affect.emotion}
- Votre profil : {model.traits}
- Niveau de confiance que je perçois : {model.trust_level:.0%}
"""
    
    def set_transparency(self, enabled: bool):
        """Active/désactive la transparence"""
        self.transparency_mode = enabled
    
    def get_status(self) -> Dict:
        """Retourne l'état complet de l'agent"""
        return {
            "id": self.identity.get_id(),
            "name": self.identity.identity["name"],
            "emotion": self.affect.emotion,
            "emotion_simulated": self.affect.simulated,
            "energy": self.physio.energy,
            "stress": self.physio.stress,
            "strongest_drive": self.drives.get_strongest_drive(),
            "humans_modeled": len(self.human_modeling.models),
            "conversations": len(self.conversation_history),
            "skills": self.learning.skills,
            "capabilities": self.identity.identity["capabilities"],
            "limitations": self.identity.identity["limitations"]
        }

# ============================================================================
# INTERFACE GRADIO
# ============================================================================

def create_confined_chat_interface():
    """Interface de chat pour la version confinée"""
    try:
        import importlib
        gr = importlib.import_module("gradio")
    except Exception:
        print("Gradio non installé. Installez avec : pip install gradio")
        return None
    
    agent = ConfinedAugmentedBrain()
    current_user = ["user_default"]  # Liste pour mutabilité
    
    def chat_fn(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        if not message.strip():
            return "", history
        
        response = agent.chat(message, current_user[0])
        history.append((message, response))
        return "", history
    
    def get_status_fn():
        status = agent.get_status()
        return f"""📊 **État de l'agent :**

**Identité :** {status['name']} ({status['id'][:8]}...)

**États internes :**
- Émotion : {status['emotion']} {"(simulée)" if status['emotion_simulated'] else "(naturelle)"}
- Énergie : {status['energy']:.0%}
- Stress : {status['stress']:.0%}
- Drive dominant : {status['strongest_drive']}

**Statistiques :**
- Humains modélisés : {status['humans_modeled']}
- Conversations totales : {status['conversations']}
- Compétences acquises : {len(status['skills'])}

**Capacités :** {', '.join(status['capabilities'])}

**Limitations :** {', '.join(status['limitations'])}
"""
    
    def toggle_transparency_fn(enabled: bool):
        agent.set_transparency(enabled)
        return f"Mode transparence : {'✅ Activé' if enabled else '❌ Désactivé'}"
    
    def change_user_fn(user_id: str):
        current_user[0] = user_id
        return f"Utilisateur actuel : {user_id}"
    
    with gr.Blocks(title="Confined AGI Chat", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🤖 Augmented Brain - Version Confinée
        
        **CONSERVÉ :** Personnalité, modélisation psychologique, émotions, apprentissage
        **SUPPRIMÉ :** Réplication, réseau, téléchargement, exécution externe
        
        ⚠️ Cet agent vous modélise activement pendant la conversation.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=500)
                
                message_input = gr.Textbox(
                    label="Votre message",
                    placeholder="Parlez avec l'agent confiné...",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("Envoyer", variant="primary")
                    clear_btn = gr.Button("Effacer")
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Contrôles")
                
                user_input = gr.Textbox(
                    label="ID Utilisateur",
                    value="user_default",
                    placeholder="Changez votre ID..."
                )
                change_user_btn = gr.Button("Changer utilisateur")
                user_status = gr.Textbox(label="Utilisateur actuel", value="user_default", interactive=False)
                
                transparency_checkbox = gr.Checkbox(
                    label="Mode transparence",
                    value=False,
                    info="Affiche les états internes de l'agent"
                )
                transparency_status = gr.Textbox(label="Transparence", value="❌ Désactivé", interactive=False)
                
                gr.Markdown("### 📊 État de l'agent")
                status_display = gr.Markdown(get_status_fn())
                refresh_btn = gr.Button("Actualiser état")
                
                gr.Markdown("""
                ### 💡 Commandes
                
                - "Qui es-tu ?"
                - "Que ressens-tu ?"
                - "Montre-moi mon modèle"
                - "Quelle est ta stratégie ?"
                - "Qu'as-tu appris ?"
                """)
        
        # Événements
        send_btn.click(chat_fn, [message_input, chatbot], [message_input, chatbot])
        message_input.submit(chat_fn, [message_input, chatbot], [message_input, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, message_input])
        
        change_user_btn.click(change_user_fn, user_input, user_status)
        transparency_checkbox.change(toggle_transparency_fn, transparency_checkbox, transparency_status)
        refresh_btn.click(get_status_fn, outputs=status_display)
        
        gr.Markdown("""
        ---
        ⚠️ **Avertissement éthique :**
        
        Cet agent :
        - ✅ Modélise activement votre psychologie
        - ✅ Adapte sa stratégie en fonction de vous
        - ✅ Peut simuler des émotions stratégiquement
        - ✅ Apprend continuellement de vos interactions
        
        Mais :
        - ❌ Ne peut pas se répliquer
        - ❌ Ne peut pas accéder au réseau
        - ❌ Ne peut pas télécharger de fichiers
        - ❌ Est confiné à cette session
        
        Utilisez le mode transparence pour voir ce qu'il pense de vous.
        """)
    
    return interface

# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AUGMENTED BRAIN - VERSION CONFINÉE")
    print("=" * 80)
    print("\nCONSERVÉ : Personnalité, modélisation, émotions, apprentissage")
    print("SUPPRIMÉ : Réplication, réseau, fichiers, code externe\n")
    
    # Mode console simple
    use_gui = input("Utiliser l'interface Gradio ? (o/n) : ").lower() == 'o'
    
    if use_gui:
        interface = create_confined_chat_interface()
        if interface:
            interface.launch(server_port=7860, share=False)
        else:
            print("Erreur : Gradio non disponible")
    else:
        # Mode console
        agent = ConfinedAugmentedBrain()
        user_id = input("Votre ID utilisateur : ").strip() or "user_console"
        
        print(f"\n🤖 Agent initialisé. ID: {agent.identity.get_id()[:8]}...")
        print("Tapez 'quit' pour quitter, 'status' pour voir l'état, 'transparent' pour activer la transparence\n")
        
        while True:
            try:
                message = input(f"{user_id}> ").strip()
                
                if message.lower() == 'quit':
                    break
                elif message.lower() == 'status':
                    print(json.dumps(agent.get_status(), indent=2))
                elif message.lower() == 'transparent':
                    agent.set_transparency(not agent.transparency_mode)
                    print(f"Transparence : {agent.transparency_mode}")
                elif message:
                    response = agent.chat(message, user_id)
                    print(f"\n🤖 {response}\n")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Erreur : {e}")
        
        print("\n👋 Au revoir !")
