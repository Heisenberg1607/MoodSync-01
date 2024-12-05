import { useState, useEffect } from "react";
import axios from "axios";

const App = () => {
  const [status, setStatus] = useState("Click to start session");
  const [batchNumber, setBatchNumber] = useState(1);
  const [dominantEmotion, setDominantEmotion] = useState(null);
  const [chatResponse, setChatResponse] = useState(null);

  const startSession = async () => {
    try {
      setStatus("Starting session...");
      const response = await axios.post("http://127.0.0.1:5000/start-session", {
        batch_number: batchNumber,
      });
      setStatus(response.data.status);

      // Start fetching the dominant emotion after session starts
      fetchDominantEmotion();
    } catch (error) {
      console.error("Error starting session:", error);
      setStatus("Failed to start session. Check backend logs.");
    }
  };

  const fetchDominantEmotion = async () => {
    try {
      setStatus("Fetching dominant emotion...");
      const interval = setInterval(async () => {
        const response = await axios.get(
          `http://127.0.0.1:5000/get-dominant-emotion?batch_number=${batchNumber}`
        );
        if (response.data.dominant_emotion) {
          setDominantEmotion(response.data.dominant_emotion);
          setStatus(`Dominant emotion: ${response.data.dominant_emotion}`);
          clearInterval(interval); // Stop polling
        }
      }, 2000); // Poll every 2 seconds
    } catch (error) {
      console.error("Error fetching dominant emotion:", error);
      setStatus("Failed to fetch dominant emotion. Check backend logs.");
    }
  };

  const fetchChatResponse = async () => {
    if (!dominantEmotion) return;
    setStatus("Fetching chatbot response...");
    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/get-chat-response",
        {
          emotion: dominantEmotion,
        }
      );
      setChatResponse(response.data.message);
      setStatus("Chatbot response received.");
    } catch (error) {
      console.error("Error fetching chatbot response:", error);
      setStatus("Failed to fetch chatbot response.");
    }
  };

  // Automatically trigger chatbot when dominantEmotion is set
  useEffect(() => {
    if (dominantEmotion) {
      fetchChatResponse();
    }
  }, [dominantEmotion]);

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>MoodSync</h1>
      <p>Status: {status}</p>
      <div style={{ marginBottom: "20px" }}>
        <label>Batch Number: </label>
        <input
          type="number"
          value={batchNumber}
          onChange={(e) => setBatchNumber(e.target.value)}
          style={{
            padding: "5px",
            fontSize: "16px",
            marginLeft: "10px",
            width: "80px",
          }}
        />
      </div>
      <button
        onClick={startSession}
        style={{
          padding: "10px 20px",
          fontSize: "16px",
          backgroundColor: "#007BFF",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
        }}
      >
        Begin Session
      </button>
      {dominantEmotion && (
        <div style={{ marginTop: "20px" }}>
          <h2>Dominant Emotion: {dominantEmotion}</h2>
        </div>
      )}
      {chatResponse && (
        <div style={{ marginTop: "20px" }}>
          <h3>Chatbot Suggestion:</h3>
          <p>{chatResponse}</p>
        </div>
      )}
    </div>
  );
};

export default App;
