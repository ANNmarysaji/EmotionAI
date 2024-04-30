import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores,gameno):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of games")
    plt.ylabel("Score")
    plt.plot(scores)
    # plt.plot(mean_scores)
    # plt.ylim(ymin=0)
    # plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    # plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    if mean_scores:
      plt.plot(mean_scores)
      if mean_scores[-1] > plt.ylim()[1]:  # Check if score is above limit
          y_pos = plt.ylim()[1] - 0.1  # Place text slightly below limit
      else:
          y_pos = mean_scores[-1]
    plt.text(len(scores)-1, y_pos, str(mean_scores[-1]))
    plt.pause(0.001)  
    if 550 <= gameno <= 560 or gameno==600:
        plt.tight_layout()
        plt.savefig(f"plot_{gameno}.png", dpi=600)
