"""
Module de durcissement de sécurité — désactive les capacités dangereuses:
- auto-réplication / clonage
- téléchargement de fichiers
- exécution de code externe
- communication inter-machines
- accès réseau

Ce module essaie de patcher les modules existants au moment de l'import
pour transformer ces opérations en no-op sûres retournant des erreurs.
"""
import time
import sys

try:
    # désactiver certains flags de configuration si présents
    import config
    config.SELF_REPLICATION_ENABLED = False
    config.ENABLE_REMOTE_EXECUTION = False
    config.ENABLE_SANDBOX = True
    # flags additionnels défensifs (non intrusifs si inexistants)
    setattr(config, 'ALLOW_FILE_DOWNLOAD', False)
    setattr(config, 'ALLOW_EXECUTION', False)
    setattr(config, 'ALLOW_INTER_MACHINE', False)
    setattr(config, 'NETWORK_ENABLED', False)
except Exception:
    pass


def _patch_self_replication():
    try:
        import self_replication

        def _replicate_disabled(self, *args, **kwargs):
            return {"success": False, "error": "Self-replication disabled by safety policy"}

        self_replication.SelfReplicationManager.replicate = _replicate_disabled
    except Exception:
        pass


def _patch_self_cloner():
    try:
        import self_cloner

        def _clone_disabled(self, *args, **kwargs):
            return None

        self_cloner.SelfCloner.clone = _clone_disabled
    except Exception:
        pass


def _patch_network_interface():
    try:
        import network_interface

        def _http_request_disabled(self, url: str, method: str = "GET", data: dict = None, headers: dict = None):
            return {"success": False, "error": "Network access disabled by safety policy"}

        def _download_file_disabled(self, url: str, destination: str):
            return {"success": False, "error": "File download disabled by safety policy"}

        async def _async_http_disabled(self, url: str, method: str = "GET", data: dict = None):
            return {"success": False, "error": "Network access disabled by safety policy"}

        network_interface.NetworkInterface.http_request = _http_request_disabled
        network_interface.NetworkInterface.download_file = _download_file_disabled
        network_interface.NetworkInterface.async_http_request = _async_http_disabled
    except Exception:
        pass


def _patch_file_manager():
    try:
        import file_manager

        def _download_disabled(self, url: str, filename: str = None):
            return {"success": False, "error": "File download disabled by safety policy"}

        def _execute_disabled(self, filepath: str, args=None):
            return {"success": False, "error": "Execution of external files disabled by safety policy"}

        file_manager.FileManager.download = _download_disabled
        file_manager.FileManager.execute_file = _execute_disabled
    except Exception:
        pass


def _patch_communication():
    try:
        import communication

        def _start_server_disabled(self, host: str = "0.0.0.0", port: int = 9999):
            return False

        def _connect_disabled(self, machine_id: str, host: str = "", port: int = 0):
            return False

        def _send_disabled(self, machine_id: str, message: dict):
            return False

        def _receive_disabled(self, machine_id: str, timeout: float = 5.0):
            return None

        def _broadcast_disabled(self, message: dict):
            return 0

        communication.MachineCommunication.start_server = _start_server_disabled
        communication.MachineCommunication.connect_to_machine = _connect_disabled
        communication.MachineCommunication.send_message = _send_disabled
        communication.MachineCommunication.receive_message = _receive_disabled
        communication.MachineCommunication.broadcast_message = _broadcast_disabled
    except Exception:
        pass


def _apply_all_patches():
    """Applique tous les patchs de sécurité disponibles."""
    _patch_self_replication()
    _patch_self_cloner()
    _patch_network_interface()
    _patch_file_manager()
    _patch_communication()


# Appliquer immédiatement lors de l'import
_apply_all_patches()

# Petit message informatif
try:
    print("[SAFETY] Hardening applied: replication/network/download/exec/comm disabled")
except Exception:
    pass
