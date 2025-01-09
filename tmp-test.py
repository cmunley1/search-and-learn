from langchain_nvidia_ai_endpoints import ChatNVIDIA

def test_batch_output():
    # Initialize the model
    llm = ChatNVIDIA(base_url="http://localhost:8000/v1", model="mistral-nemo-12b-instruct")
    
    # Test batch with simple questions
    test_inputs = [
        "What's 2+2?",
    ]
    
    # Get batch results
    results = llm.batch(test_inputs)
    print(f"RESULTS: {results}") 
    # Print results with clear separation
    print("\n=== Batch Output Results ===")
    for i, result in enumerate(results):
        print(f"\nInput {i+1}: {test_inputs[i]}")
        print(f"Output {i+1}: {result.content}")
        print(f"Stop Reason: {result.additional_kwargs.get('stop_reason', 'None')}")
        print("-" * 50)

if __name__ == "__main__":
    test_batch_output()

