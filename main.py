from asr import capture_audio
from llm import ConversationalAgent
from tts import text_to_speech

def main():
    print("=====================================================")
    print("Initializing Voice-Based Conversational AI System...")
    print("=====================================================")
    agent = ConversationalAgent()
    print("\nSystem ready! Speak into your microphone.")
    print("Say 'exit', 'quit', or 'stop' to end the conversation.")
    
    while True:
        user_input = capture_audio()
        
        if user_input:
            if user_input.lower() in ['exit', 'quit', 'stop']:
                text_to_speech("Goodbye!")
                break
                
            response = agent.generate_response(user_input)
            text_to_speech(response)

if __name__ == "__main__":
    main()
