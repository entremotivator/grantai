import streamlit as st
import requests
import json
import time
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import queue

# OllamaAPI class remains unchanged from the original code

class OllamaAPI:
    def __init__(self, base_url: str, username: str, password: str, timeout: int = 90):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.auth = (self.username, self.password)
        return session

    def _make_request(self, payload: Dict) -> Dict:
        """Make request with timeout handling."""
        headers = {"Content-Type": "application/json"}
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions", headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def generate_with_backup(self, messages: List[Dict], temperature: float = 0.7) -> Tuple[str, Dict]:
        """Generate response with backup settings if primary fails."""
        payloads = [
            {
                "model": "llama3.2",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000,
                "stream": False,
            },
            # Backup settings with lower complexity
            {
                "model": "llama3.2",
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 1000,
                "stream": False,
            },
        ]
        for payload in payloads:
            try:
                response = self._make_request(payload)
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content, response
            except Exception as e:
                last_error = str(e)
                continue
        raise Exception(f"All generation attempts failed. Last error: {last_error}")


class PostGenerator:
    def __init__(self):
        self.api = OllamaAPI(
            base_url=st.session_state.api_url,
            username=st.session_state.username,
            password=st.session_state.password,
        )

    def generate_grant_section(self, grant_details: Dict[str, str]) -> str:
        """Generate a grant section based on provided details."""
        messages = [
            {"role": "system", "content": "You are an expert grant writer."},
            {"role": "user", "content": f"Create a grant section using these details: {json.dumps(grant_details, indent=2)}"},
        ]
        try:
            content, response = self.api.generate_with_backup(messages, st.session_state.temperature)
            return content
        except Exception as e:
            st.error(f"Error generating grant section: {str(e)}")
            return "Unable to generate grant section. Please try again."


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "messages": [],
        "api_url": "https://theaisource-u29564.vm.elestio.app:57987",
        "username": "root",
        "password": "eZfLK3X4-SX0i-UmgUBe6E",
        "selected_model": "llama3.2",
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_grant_interface():
    """Create an interface for generating grant sections."""
    st.markdown("### 🎯 Grant Section Generator")
    col1, col2 = st.columns(2)
    with col1:
        project_name = st.text_input("Project Name")
        organization_name = st.text_input("Organization Name")
    with col2:
        funding_goal = st.number_input("Funding Goal ($)", min_value=0)
        deadline = st.date_input("Submission Deadline")

    project_description = st.text_area("Project Description", height=150)
    if st.button("Generate Grant Section"):
        with st.spinner("Generating your grant section..."):
            try:
                generator = PostGenerator()
                grant_details = {
                    "project_name": project_name,
                    "organization_name": organization_name,
                    "funding_goal": funding_goal,
                    "deadline": deadline.isoformat(),
                    "project_description": project_description,
                }
                grant_section = generator.generate_grant_section(grant_details)
                display_generated_grant_section(grant_section)
            except Exception as e:
                st.error(f"Failed to generate grant section: {str(e)}")


def display_generated_grant_section(grant_section: str):
    """Display the generated grant section."""
    st.markdown("### 📝 Generated Grant Section")
    st.text_area("Grant Section Content", grant_section, height=200)


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Professional Social Media and Grant Generator",
        page_icon="🌐",
        layout="wide",
    )
    init_session_state()
    st.title("🌐 Professional Social Media & Grant Generator")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Create Post", "Grant Generator", "Post History"])
    
    with tab1:
        create_post_interface()  # Existing post generator interface
    
    with tab2:
        create_grant_interface()  # New grant generator interface
    
    with tab3:
        display_post_history()  # Existing post history interface


if __name__ == "__main__":
    main()
