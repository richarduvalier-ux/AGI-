"""Script d'exécution sécurisé pour la version confinée de l'agent.
Il importe `safety_hardening` pour s'assurer que les capacités à risque
restent neutralisées, puis lance un REPL console pour l'agent confiné.
"""

# Appliquer le durcissement dès le départ
try:
    import safety_hardening
except Exception:
    pass

from confined_agent import ConfinedAugmentedBrain
import json


def main():
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


if __name__ == '__main__':
    main()
