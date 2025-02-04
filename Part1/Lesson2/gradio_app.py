from fastai.vision.all import load_learner
import gradio as gr

# Define function for model namespace
def is_cat(x): return x[0].isupper() 

# Load the trained model
learn = load_learner(r'Part1\Lesson2\model.pkl')

def classify_image(image):
    # Get prediction
    pred, pred_idx, probs = learn.predict(image)
    return f"Prediction: {pred}\nCat Probability: {probs[1]:.4f}\nNot Cat Probability: {probs[0]:.4f}"

# Create Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(width=512, height=512),
    outputs=gr.Textbox(),
    title="Cat vs Not Cat Classifier",
    description="Upload an image to check if it's a cat or not",
    live=True  # Enable automatic prediction on upload 
)

# Launch the app
demo.launch()