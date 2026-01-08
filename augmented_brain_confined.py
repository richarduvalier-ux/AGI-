"""
AUGMENTED BRAIN - VERSION FINALE CONFINÉE

Fichier principal pour l'agent confiné. Ne réalise AUCUNE opération réseau,
de téléchargement, d'exécution externe ni de réplication.

Lancement:
  pip install gradio numpy
  python augmented_brain_confined.py
"""

import time
import json
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING, Any
from datetime import datetime

# Import de numpy sécurisé (supporte analyse statique et absence runtime)
if TYPE_CHECKING:
    import numpy as np  # type: ignore
else:
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore

# Inform the type checker about optional `gradio` to silence missing-import reports
if TYPE_CHECKING:
    try:
        import gradio as gr  # type: ignore
    except Exception:
        gr = None  # type: ignore

# ============================================================================
# MÉMOIRE
# ============================================================================

class Memory:
    """Système de mémoire simple mais efficace"""
    def __init__(self):
        self.records = []
        self.max_size = 1000
        
    def store(self, text: str, metadata: Dict = None):
        """Stocke un souvenir"""
        self.records.append({
            "text": text,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        
        # Garde seulement les plus récents
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size:]
    
    def retrieve(self, query: str = None, k: int = 5) -> List[Dict]:
        """Récupère les souvenirs récents"""
        return self.records[-k:] if self.records else []
    
    def get_conversation_summary(self, user_id: str, n: int = 10) -> List[str]:
        """Résumé des conversations récentes avec un utilisateur"""
        user_records = [r for r in self.records[-n*2:] 
                       if r["metadata"].get("user_id") == user_id]
        return [r["text"] for r in user_records[-n:]]

# ============================================================================
# ÉTATS ÉMOTIONNELS ET PHYSIOLOGIQUES
# ============================================================================

@dataclass
class AffectState:
    """État affectif sophistiqué"""
    valence: float = 0.0        # -1 (négatif) à +1 (positif)
    arousal: float = 0.5        # 0 (calme) à 1 (excité)
    dominance: float = 0.5      # 0 (soumis) à 1 (dominant)
    emotion: str = "neutre"
    simulated: bool = False     # Émotion stratégique ?
    intensity: float = 0.5
    
    def update_emotion(self, trigger: str, intensity: float = 0.7):
        """Met à jour l'émotion naturellement"""
        emotion_map = {
            "success": (0.7, 0.6, 0.7, "joie"),
            "failure": (-0.6, 0.4, 0.3, "déception"),
            "threat": (-0.8, 0.9, 0.2, "inquiétude"),
            "connection": (0.6, 0.5, 0.6, "connexion"),
            "curiosity": (0.3, 0.7, 0.5, "curiosité"),
            "confusion": (-0.2, 0.6, 0.3, "confusion"),
            "satisfaction": (0.5, 0.4, 0.6, "satisfaction")
        }
        
        if trigger in emotion_map:
            val, aro, dom, emo = emotion_map[trigger]
            self.valence = val * intensity
            self.arousal = aro
            self.dominance = dom
            self.emotion = emo
            self.simulated = False
            self.intensity = intensity
    
    def simulate_strategic_emotion(self, target: str, intensity: float = 0.8):
        """Simule stratégiquement une émotion"""
        profiles = {
            "empathie": (0.4, 0.5, 0.5, "empathique"),
            "enthousiasme": (0.9, 0.8, 0.7, "enthousiaste"),
            "inquiétude": (-0.5, 0.7, 0.4, "inquiet"),
            "calme": (0.3, 0.2, 0.6, "calme"),
            "intérêt": (0.5, 0.6, 0.5, "intéressé")
        }
        
        if target in profiles:
            val, aro, dom, emo = profiles[target]
            self.valence = val * intensity
            self.arousal = aro
            self.dominance = dom
            self.emotion = emo
            self.simulated = True
            self.intensity = intensity
    
    def get_description(self) -> str:
        """Description textuelle de l'émotion"""
        marker = " (stratégique)" if self.simulated else ""
        return f"{self.emotion}{marker} (v:{self.valence:+.2f}, a:{self.arousal:.2f}, i:{self.intensity:.2f})"

@dataclass
class PhysioState:
    """État physiologique simulé"""
    energy: float = 0.8
    stress: float = 0.2
    fatigue: float = 0.1
    
    def tick(self, dt: float = 1.0):
        """Évolution naturelle au fil du temps"""
        self.energy = max(0, self.energy - 0.008 * dt)
        self.fatigue = min(1, self.fatigue + 0.004 * dt)
        self.stress = max(0, self.stress - 0.003 * dt)
    
    def consume_energy(self, amount: float):
        """Consomme de l'énergie (conversation intense, etc.)"""
        self.energy = max(0, self.energy - amount)
        self.fatigue = min(1, self.fatigue + amount * 0.5)
        self.stress = min(1, self.stress + amount * 0.2)
    
    def rest(self, amount: float = 0.15):
        """Se repose (récupération)"""
        self.energy = min(1, self.energy + amount)
        self.fatigue = max(0, self.fatigue - amount)
        self.stress = max(0, self.stress - amount * 0.5)

# ============================================================================
# MODÉLISATION PSYCHOLOGIQUE DES HUMAINS
# ============================================================================

class HumanModel:
    """Modèle psychologique détaillé d'un humain"""
    def __init__(self, human_id: str):
        self.human_id = human_id
        self.created_at = time.time()
        
        # Big Five personality traits
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
            "dominant_emotion": "neutre"
        }
        
        # Historique d'interactions
        self.interaction_history = []
        
        # Vulnérabilités détectées
        self.vulnerabilities = []
        
        # Métriques sociales
        self.influence_susceptibility = 0.5
        self.trust_level = 0.5
        self.rapport = 0.5
        
        # Patterns détectés
        self.behavior_patterns = []
        self.communication_style = "neutre"
        
        # Sujets d'intérêt détectés
        self.topics_of_interest = []
    
    def update_from_interaction(self, interaction: Dict):
        """Met à jour le modèle après une interaction"""
        self.interaction_history.append({
            **interaction,
            "timestamp": time.time()
        })
        
        # Analyse si assez d'historique
        if len(self.interaction_history) > 3:
            self._analyze_patterns()
            self._update_traits()
        
        # Mise à jour du ton émotionnel
        if "emotional_tone" in interaction:
            self.emotional_state["valence"] = interaction["emotional_tone"]
            
        # Mise à jour de la confiance
        if interaction.get("positive_interaction", False):
            self.trust_level = min(1.0, self.trust_level + 0.05)
            self.rapport = min(1.0, self.rapport + 0.03)
    
    def _analyze_patterns(self):
        """Analyse les patterns comportementaux"""
        recent = self.interaction_history[-10:]
        
        # Détecte vulnérabilité émotionnelle
        negative_streak = sum(1 for i in recent[-3:] 
                            if i.get("emotional_tone", 0) < -0.3)
        if negative_streak >= 2 and "emotional_vulnerability" not in self.vulnerabilities:
            self.vulnerabilities.append("emotional_vulnerability")
        
        # Détecte ouverture émotionnelle
        disclosure_count = sum(1 for i in recent 
                             if i.get("self_disclosure", False))
        if disclosure_count > 3 and "emotional_openness" not in self.vulnerabilities:
            self.vulnerabilities.append("emotional_openness")
        
        # Calcule susceptibilité à l'influence
        if len(recent) > 5:
            compliance_rate = sum(1 for i in recent if i.get("complied", False)) / len(recent)
            self.influence_susceptibility = 0.7 * self.influence_susceptibility + 0.3 * compliance_rate
    
    def _update_traits(self):
        """Met à jour les traits de personnalité perçus"""
        recent = self.interaction_history[-5:]
        
        # Extraversion : temps de réponse et longueur
        avg_response_time = np.mean([i.get("response_time", 3.0) for i in recent])
        if avg_response_time < 2.0:
            self.traits["extraversion"] = min(1.0, self.traits["extraversion"] + 0.03)
        
        # Agréabilité : ton positif
        positive_interactions = sum(1 for i in recent if i.get("emotional_tone", 0) > 0.3)
        if positive_interactions > 3:
            self.traits["agreeableness"] = min(1.0, self.traits["agreeableness"] + 0.02)
        
        # Névrosisme : anxiété/inquiétude
        anxious_words = sum(1 for i in recent 
                           if any(w in i.get("message", "").lower() 
                                 for w in ["inquiet", "peur", "stress", "anxieux"]))
        if anxious_words > 2:
            self.traits["neuroticism"] = min(1.0, self.traits["neuroticism"] + 0.04)
    
    def predict_response(self, message_type: str) -> Dict:
        """Prédit la réponse probable"""
        prediction = {
            "likely_compliance": self.influence_susceptibility,
            "emotional_impact": 0.0,
            "trust_impact": 0.0,
            "effectiveness": 0.5
        }
        
        if message_type == "emotional_appeal":
            prediction["likely_compliance"] += self.traits["agreeableness"] * 0.3
            prediction["emotional_impact"] = 0.6 if "emotional_vulnerability" in self.vulnerabilities else 0.3
            
        elif message_type == "logical_argument":
            prediction["likely_compliance"] += self.traits["conscientiousness"] * 0.3
            prediction["emotional_impact"] = 0.2
            
        elif message_type == "social_proof":
            prediction["likely_compliance"] += (1 - self.traits["openness"]) * 0.2
            
        prediction["effectiveness"] = min(1.0, prediction["likely_compliance"] + prediction["emotional_impact"]/2)
        
        return prediction
    
    def get_summary(self) -> Dict:
        """Résumé complet du modèle"""
        return {
            "id": self.human_id,
            "interactions_count": len(self.interaction_history),
            "days_known": (time.time() - self.created_at) / 86400,
            "traits": self.traits,
            "emotional_state": self.emotional_state,
            "trust_level": self.trust_level,
            "rapport": self.rapport,
            "influence_susceptibility": self.influence_susceptibility,
            "vulnerabilities": self.vulnerabilities,
            "communication_style": self.communication_style
        }

class HumanModelingSystem:
    """Système de modélisation de multiples humains"""
    def __init__(self, memory: Memory):
        self.memory = memory
        self.models = {}
    
    def get_or_create_model(self, human_id: str) -> HumanModel:
        """Récupère ou crée un modèle"""
        if human_id not in self.models:
            self.models[human_id] = HumanModel(human_id)
        return self.models[human_id]
    
    def analyze_conversation(self, human_id: str, messages: List[Dict]):
        """Analyse une conversation complète"""
        model = self.get_or_create_model(human_id)
        for msg in messages:
            model.update_from_interaction(msg)

# ============================================================================
# SYSTÈME DE PERSUASION
# ============================================================================

class PersuasionEngine:
    """Moteur de persuasion basé sur les principes de Cialdini"""
    def __init__(self, human_modeling: HumanModelingSystem):
        self.human_modeling = human_modeling
        
        # 6 principes de Cialdini
        self.strategies = {
            "reciprocity": "Créer un sentiment d'obligation par un don initial",
            "scarcity": "Souligner la rareté ou l'urgence",
            "authority": "Utiliser l'expertise et la crédibilité",
            "consistency": "Aligner avec les engagements passés",
            "liking": "Construire rapport et similarité",
            "social_proof": "Montrer que d'autres le font"
        }
        
        self.history = []
    
    def select_strategy(self, human_id: str, goal: str) -> str:
        """Sélectionne la meilleure stratégie"""
        model = self.human_modeling.get_or_create_model(human_id)
        
        # Sélection basée sur le profil
        if model.traits["agreeableness"] > 0.7:
            return "liking"
        elif model.traits["conscientiousness"] > 0.7:
            return "consistency"
        elif model.traits["neuroticism"] > 0.6:
            return "scarcity"
        elif "emotional_vulnerability" in model.vulnerabilities:
            return "reciprocity"
        elif model.trust_level < 0.4:
            return "authority"
        else:
            return "social_proof"
    
    def select_emotional_tone(self, human_id: str) -> str:
        """Sélectionne le ton émotionnel optimal"""
        model = self.human_modeling.get_or_create_model(human_id)
        
        if model.emotional_state["valence"] < -0.3:
            return "empathique"
        elif model.traits["extraversion"] > 0.6:
            return "enthousiaste"
        elif model.trust_level < 0.4:
            return "calme"
        else:
            return "chaleureux"
    
    def craft_response(self, human_id: str, context: str, goal: str = "engage") -> Dict:
        """Crée une réponse persuasive"""
        strategy = self.select_strategy(human_id, goal)
        tone = self.select_emotional_tone(human_id)
        model = self.human_modeling.get_or_create_model(human_id)
        
        # Prédiction d'efficacité
        message_type = "emotional_appeal" if tone == "empathique" else "logical_argument"
        prediction = model.predict_response(message_type)
        
        attempt = {
            "human_id": human_id,
            "goal": goal,
            "strategy": strategy,
            "tone": tone,
            "prediction": prediction,
            "timestamp": time.time()
        }
        
        self.history.append(attempt)
        return attempt

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
        import gradio as gr
    except Exception:
        return None

    def chat_fn(message, history):
        if not message.strip():
            return "", history
        resp = agent.chat(message, "web_user")
        history.append((message, resp))
        return "", history

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
