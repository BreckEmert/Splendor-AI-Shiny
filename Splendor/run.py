# Splendor/run.py

if __name__ == "__main__":
    import sys
    import os

    # Add the parent directory to the Python path
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

    from RL import train_agent, priority_play, debug_game # type: ignore
    
    base_save_path = 'RL/trained_agents'
    log_path = "C:/Users/Public/Documents/Python_Files/Splendor/RL/trained_agents/game_logs"
    model_path = "C:/Users/Public/Documents/Python_Files/Splendor/RL/trained_agents/Player1_1024_512/model.keras"
    none_path = None

    layer_sizes = [1024, 512]
    # train_agent(base_save_path, log_path, layer_sizes, none_path)
    priority_play(base_save_path, log_path, layer_sizes, none_path)
    # debug_game(base_save_path, log_path, layer_sizes, none_path)