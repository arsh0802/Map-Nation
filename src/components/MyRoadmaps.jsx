import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import "./MyRoadmaps.css";

function FetchLearning({ setFullData }) {
  const [languageName, setLanguageName] = useState("");
  const [days, setDays] = useState("");
  const [hours, setHours] = useState("");
  const [preparingFor, setPreparingFor] = useState("");
  const [loading, setLoading] = useState(false); // State for loader
  const navigate = useNavigate(); // Use navigate hook

  const handleFetchTopics = async () => {
    if (!languageName || !days || !hours || !preparingFor) {
      alert("Please fill in all required fields.");
      return;
    }

    try {
      setLoading(true); // Show loader
      console.log("Sending API call");
      const response = await axios.post("http://localhost:5000/api/roadmaps/generate-roadmap", {
        prompt: `Provide a learning plan for ${languageName} for ${days} days where I can spend ${hours} hours as I am preparing for ${preparingFor}.`
      });

      setFullData(response.data); // Store fetched data in state
      navigate("/topics"); // Navigate to LanguageTopics page after data is fetched
    } catch (error) {
      console.error("Error fetching topics:", error);
    } finally {
      setLoading(false); // Hide loader after API call completes
    }
  };

  return (
    <div className="container">
      <div className="form-container">
        <h1>MAP&nbsp;&nbsp;NATION</h1>
        <input
          type="text"
          value={languageName}
          onChange={(e) => setLanguageName(e.target.value)}
          placeholder="Enter Language Name"
          required
        />
        <input
          type="number"
          value={days}
          onChange={(e) => setDays(e.target.value)}
          placeholder="Enter Number of Days"
          required
        />
        <input
          type="number"
          value={hours}
          onChange={(e) => setHours(e.target.value)}
          placeholder="Enter Number of Hours a day"
          required
        />
        <input
          type="text"
          value={preparingFor}
          onChange={(e) => setPreparingFor(e.target.value)}
          placeholder="What are you preparing for"
          required
        />
        <button onClick={handleFetchTopics} disabled={loading}>
          {loading ? "Loading..." : "Let's Go  !!"}
        </button>
        {loading && <div className="loader"></div>} {/* Loader element */}
      </div>
      <div className="threeDView">
        <iframe
          src="https://my.spline.design/littlerobot-8a23c1e322dfee99930bc1e0a83456bf/"
          frameBorder="0"
          width="100%"
          height="100%"
          allow="fullscreen"
          title="3D Model"
        ></iframe>
      </div>
    </div>
  );
}

export default FetchLearning;
