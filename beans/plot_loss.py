import json
import matplotlib.pyplot as plt
import argparse

def plot_training_logs(log_file, output_file="loss_plot.png"):
    # Read log file
    with open(log_file, 'r') as file:
        lines = file.readlines()
            
        # Skip the first line and the last two lines (non-JSON content)
        lines = lines[1:-2]

        lines = [line.replace("'", '"') for line in lines]
            
        # Parse the JSON content from the remaining lines
        logs = [json.loads(line.strip()) for line in lines if line.strip()]
        
        if not logs:
            print(f"Error: The log file {log_file} is empty or does not contain valid JSON.")
            return
    
    # Extract losses
    epochs = [log['epoch'] for log in logs]
    train_losses = [log['train']['loss'] for log in logs]
    valid_losses = [log['valid']['loss'] for log in logs]
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs, valid_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    
    # Save the plot as a PNG file
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Plot training and validation losses from log file.")
    parser.add_argument("-l", "--log_file", required=True, help="Path to the log file.")
    parser.add_argument("-o", "--output_file", default="loss_plot.png", help="Path to save the output plot (default: loss_plot.png).")
    
    args = parser.parse_args()
    
    # Plot the training logs
    plot_training_logs(args.log_file, args.output_file)
