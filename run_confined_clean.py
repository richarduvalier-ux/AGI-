"""Runner propre pour démarrer l'agent confiné en mode console.
Utilise `ConfinedAugmentedBrain` depuis `confined_agent.py`.
"""

from confined_agent import ConfinedAugmentedBrain
import json
import sys


def run_session(commands=None):
    agent = ConfinedAugmentedBrain()
    if commands is None:
        # mode interactif
        user_id = input("Votre ID utilisateur : ").strip() or "user_console"
        print(f"\n🤖 Agent initialisé. ID: {agent.identity.get_id()[:8]}...")
        print("Tapez 'quit' pour quitter, 'status' pour voir l'état, 'transparent' pour activer la transparence\n")
        while True:
            try:
                message = input(f"{user_id}> ").strip()
                if not message:
                    continue
                if message.lower() == 'quit':
                    break
                elif message.lower() == 'status':
                    print(json.dumps(agent.get_status(), indent=2))
                elif message.lower() == 'transparent':
                    agent.set_transparency(not agent.transparency_mode)
                    print(f"Transparence : {agent.transparency_mode}")
                else:
                    print('\n🤖', agent.chat(message, user_id), '\n')
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Erreur : {e}")
    else:
        # mode scripté: commands is a list of lines to send as input
        user_id = commands.pop(0).strip() or "user_console"
        outputs = []
        for message in commands:
            if message.lower() == 'quit':
                break
            if message.lower() == 'status':
                outputs.append(json.dumps(agent.get_status(), indent=2))
            elif message.lower() == 'transparent':
                agent.set_transparency(not agent.transparency_mode)
                outputs.append(f"Transparence : {agent.transparency_mode}")
            else:
                outputs.append(agent.chat(message, user_id))
        return outputs


if __name__ == '__main__':
    # Si des lignes sont fournies via stdin, on les lit et exécute en mode scripté
    if not sys.stdin.isatty():
        data = sys.stdin.read().splitlines()
        if data:
            out = run_session(data)
            for o in out:
                print(o)
            sys.exit(0)
    run_session()
