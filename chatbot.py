import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Local model directories (same as in training script)
LOCAL_BASE_MODEL_DIR = "./local_base_model"
FINETUNED_MODEL_DIR = "./qwen_math_finetuned"

class QwenMathChatbot:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
    def load_model(self):
        """Load the fine-tuned model from local directories"""
        try:
            print("🔍 Loading models from local directories...")
            
            # Check if local base model exists
            if not os.path.exists(LOCAL_BASE_MODEL_DIR):
                print(f"❌ Base model directory not found: {LOCAL_BASE_MODEL_DIR}")
                print("💡 Please run the training script first to download the base model locally.")
                return False
                
            # Check if fine-tuned model exists
            if not os.path.exists(FINETUNED_MODEL_DIR):
                print(f"❌ Fine-tuned model directory not found: {FINETUNED_MODEL_DIR}")
                print("💡 Please run the training script first to create the fine-tuned model.")
                return False
            
            # Load model info if available
            model_info_path = os.path.join(FINETUNED_MODEL_DIR, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                print(f"📋 Model info: {model_info}")
            
            print(f"📂 Loading base model from: {LOCAL_BASE_MODEL_DIR}")
            
            # Load base model from local directory
            base_model = AutoModelForCausalLM.from_pretrained(
                LOCAL_BASE_MODEL_DIR,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"🎯 Loading fine-tuned adapter from: {FINETUNED_MODEL_DIR}")
            
            # Load the fine-tuned PEFT model
            self.model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_DIR)
            
            # Load tokenizer from fine-tuned directory (it should have the tokenizer files)
            if os.path.exists(os.path.join(FINETUNED_MODEL_DIR, "tokenizer.json")) or \
               os.path.exists(os.path.join(FINETUNED_MODEL_DIR, "tokenizer_config.json")):
                self.tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
                print("📝 Loaded tokenizer from fine-tuned model directory")
            else:
                # Fallback to base model tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE_MODEL_DIR)
                print("📝 Loaded tokenizer from base model directory")
            
            # Set pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✅ Model and tokenizer loaded successfully!")
            print(f"🧠 Model loaded on: {next(self.model.parameters()).device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_response(self, user_input, max_new_tokens=200, temperature=0.7):
        """Generate response to user input"""
        try:
            # Create messages for chat template
            messages = [
                {"role": "user", "content": user_input}
            ]
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode only the new tokens (response)
            response_tokens = outputs[0][len(inputs.input_ids[0]):]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def chat_interactive(self):
        """Start interactive chat session"""
        print("\n🤖 Fine-tuned Qwen Math Chatbot")
        print("=" * 50)
        print("✨ This chatbot has been fine-tuned on math problems!")
        print("💡 Try asking math questions, equations, or problem-solving tasks.")
        print("📝 Commands: 'exit' to quit, 'reset' to clear history, 'save' to save conversation")
        print("=" * 50)
        
        if not self.load_model():
            print("❌ Failed to load model. Please ensure you have:")
            print("   1. Run the training script successfully")
            print("   2. Base model downloaded to ./local_base_model/")
            print("   3. Fine-tuned model saved to ./qwen_math_finetuned/")
            return
        
        print("\n💬 Chat started! Ask me any math question...")
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye! Thanks for chatting!")
                    break
                elif user_input.lower() == 'reset':
                    self.conversation_history = []
                    print("🔄 Conversation history cleared!")
                    continue
                elif user_input.lower() == 'save':
                    self.save_conversation()
                    continue
                elif user_input.lower() in ['help', '?']:
                    self.show_help()
                    continue
                elif not user_input:
                    print("❓ Please enter a question or command.")
                    continue
                
                # Generate and display response
                print("🤖 Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
                # Add to conversation history
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": response
                })
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue
    
    def save_conversation(self):
        """Save conversation to file"""
        if not self.conversation_history:
            print("📝 No conversation to save.")
            return
        
        filename = f"conversation_{len(self.conversation_history)}_exchanges.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("🤖 Qwen Math Chatbot Conversation\n")
                f.write("=" * 50 + "\n\n")
                
                for i, exchange in enumerate(self.conversation_history, 1):
                    f.write(f"Exchange {i}:\n")
                    f.write(f"👤 User: {exchange['user']}\n")
                    f.write(f"🤖 Assistant: {exchange['assistant']}\n\n")
            
            print(f"💾 Conversation saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving conversation: {str(e)}")
    
    def show_help(self):
        """Show help information"""
        print("\n📚 Available commands:")
        print("• 'exit' or 'quit' - End the chat")
        print("• 'reset' - Clear conversation history") 
        print("• 'save' - Save conversation to file")
        print("• 'help' or '?' - Show this help")
        print("\n💡 Math question examples:")
        print("• Solve: 2x + 5 = 15")
        print("• What is the derivative of x^2 + 3x?")
        print("• Calculate the area of a circle with radius 5")
        print("• Factor: x^2 - 5x + 6")
    
    def run_quick_tests(self):
        """Run some quick tests to verify the model works"""
        print("🧪 Running Quick Tests")
        print("=" * 30)
        
        if not self.load_model():
            print("❌ Cannot run tests - model loading failed.")
            return
        
        test_questions = [
            "Solve: 2x + 4 = 10",
            "What is 15 + 25?",
            "Factor: x^2 - 4",
            "What is the square root of 144?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔍 Test {i}: {question}")
            response = self.generate_response(question, max_new_tokens=512)
            print(f"🤖 Response: {response}")
            print("-" * 30)
        
        print("\n✅ Quick tests completed!")

def check_local_setup():
    """Check if local model setup is complete"""
    print("🔍 Checking local model setup...")
    
    base_exists = os.path.exists(LOCAL_BASE_MODEL_DIR)
    ft_exists = os.path.exists(FINETUNED_MODEL_DIR)
    
    print(f"📁 Base model directory ({LOCAL_BASE_MODEL_DIR}): {'✅ Found' if base_exists else '❌ Missing'}")
    print(f"📁 Fine-tuned model directory ({FINETUNED_MODEL_DIR}): {'✅ Found' if ft_exists else '❌ Missing'}")
    
    if base_exists:
        base_files = os.listdir(LOCAL_BASE_MODEL_DIR)
        print(f"   Contains {len(base_files)} files")
    
    if ft_exists:
        ft_files = os.listdir(FINETUNED_MODEL_DIR)
        print(f"   Contains {len(ft_files)} files")
        
        # Check for key files
        key_files = ["adapter_config.json", "adapter_model.bin", "tokenizer_config.json"]
        for key_file in key_files:
            if key_file in ft_files:
                print(f"   ✅ {key_file}")
            else:
                print(f"   ❓ {key_file} (might be missing)")
    
    if not base_exists or not ft_exists:
        print("\n💡 Setup Instructions:")
        print("1. Run the training script first: python paste.py")
        print("2. Wait for training to complete")
        print("3. Then run this chat script")
    
    return base_exists and ft_exists

def main():
    """Main function"""
    print("🚀 Choose an option:")
    print("1. Start interactive chat")
    print("2. Run quick tests")
    print("3. Check local setup")
    
    try:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            chatbot = QwenMathChatbot()
            chatbot.chat_interactive()
        elif choice == "2":
            chatbot = QwenMathChatbot()
            chatbot.run_quick_tests()
        elif choice == "3":
            check_local_setup()
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()