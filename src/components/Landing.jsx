import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Landing.css'; // Import the CSS file for styling

const LandingPage = () => {
  const navigate = useNavigate(); // Hook to programmatically navigate

  const handleLoginClick = () => {
    navigate('/login'); // Navigate to login page  
  };
  const handleSignUpClick = () => {
    navigate('/register'); // Navigate to login page  
  };
  return (
    <div className="landing-container">
      <nav className="navbar">
        <div className="logo">
          <img src="Vector(2).png" alt="Logo" className="logo-image" /> {/* Replace with your logo */}
        </div>
        <ul className="nav-links">
          <li><a href="#">Home</a></li>
          <li><a href="#">About us</a></li>
          <li><a href="#">Contact us</a></li>
        </ul>
        <div className="auth-buttons">
          <button className="login-btn" onClick={handleLoginClick}>Log in</button>
          <button className="signup-btn" onClick={handleSignUpClick}>Sign up</button>
        </div>
      </nav>

      <div className="main-content">
        <div className="left-content">
          <h1 className="main-title">Map Nation</h1>
          <p className="subtitle">
            Design your personalized learning journey with AI-powered roadmaps tailored to your goals, timelines, and language preferences.
          </p>
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
    </div>
  );
};

export default LandingPage;
