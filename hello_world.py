import streamlit as st

# Set page title
st.set_page_config(page_title="Hello World App", page_icon="👋")

# Main title
st.title("Hello World! 👋")

# Add some basic text
st.write("Welcome to my first Streamlit app!")

# Add a subheader
st.subheader("About this app")
st.write("This is a simple Hello World application built with Streamlit.")

# Add an interactive element
name = st.text_input("What's your name?", placeholder="Enter your name here")

if name:
    st.write(f"Hello, {name}! Nice to meet you! 🎉")

# Add a button
if st.button("Click me!"):
    st.balloons()
    st.success("Thanks for clicking the button!")

# Add some info in the sidebar
st.sidebar.title("Navigation")
st.sidebar.write("This is the sidebar")
st.sidebar.info("You can add navigation items here")

# Add footer info
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")