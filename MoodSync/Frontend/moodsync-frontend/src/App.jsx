import { useState } from "react";
import axios from "axios";

const App = () => {
  const [status, setStatus] = useState("Click to start session");
  const [batchNumber, setBatchNumber] = useState(1);
  const [dominantEmotion, setDominantEmotion] = useState(null);

  const startSession = async () => {
    try {
      setStatus("Starting session...");

      // Call the start-session API
      const response = await axios.post("http://127.0.0.1:5000/start-session", {
        batch_number: batchNumber, // Pass the batch number
      });

      // Update the status based on the response
      setStatus(response.data.status);
      setTimeout(() => {
        fetchDominantEmotion(); // Call the fetchDominantEmotion function
      }, 5000);
    } catch (error) {
      console.error("Error starting session:", error);
      setStatus("Failed to start session. Check backend logs.");
    }
  };

  const fetchDominantEmotion = async () => {
    try {
      setStatus("Fetching dominant emotion...");

      // Call the get-dominant-emotion API
      const response = await axios.get(
        `http://127.0.0.1:5000/get-dominant-emotion?batch_number=${batchNumber}`
      );

      // Update the state with the dominant emotion
      setDominantEmotion(response.data.dominant_emotion);
      setStatus(`Dominant emotion: ${response.data.dominant_emotion}`);
    } catch (error) {
      console.error("Error fetching dominant emotion:", error);
      setStatus("Failed to fetch dominant emotion. Check backend logs.");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>MoodSync</h1>
      <p>Status: {status}</p>

      {/* Input for batch number */}
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

      {/* Button to start the session */}
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
    </div>
  );
};

export default App;
