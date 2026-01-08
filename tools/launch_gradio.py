from augmented_brain_confined import ConfinedAugmentedBrain, create_interface

if __name__ == '__main__':
    agent = ConfinedAugmentedBrain()
    interface = create_interface(agent)
    if interface is None:
        print('Gradio interface not available (gradio not installed)')
    else:
        # prevent_thread_lock ensures launch doesn't block the process
        interface.launch(server_name='0.0.0.0', server_port=7860, share=False, prevent_thread_lock=True)
