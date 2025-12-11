
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def inspect_tags(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags()['scalars']
    print(f"--- Tags in {log_dir} ---")
    print(tags)
    
    if 'F1/val' in tags:
        events = ea.Scalars('F1/val')
        # Print all step-value pairs
        data = [(e.step, e.value) for e in events]
        print("\n--- F1/val Values (Step, Value) ---")
        for step, val in data:
            print(f"Step {step}: {val:.4f}")
            
        # Print Max
        max_pair = max(data, key=lambda x: x[1])
        print(f"\nMAX: Step {max_pair[0]} = {max_pair[1]:.5f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_tags(sys.argv[1])
