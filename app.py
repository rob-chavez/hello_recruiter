
import gradio as gr
from Me import Me


if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()

    