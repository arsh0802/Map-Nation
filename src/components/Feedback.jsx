import React, { useState } from 'react';
import './Feedback.css';

const Feedback = () => {
  const [step, setStep] = useState(1);
  const [submitted, setSubmitted] = useState(false);
  const [showSubmit, setShowSubmit] = useState(false);
  const [animateClass, setAnimateClass] = useState(''); // To handle the slide class
  const [feedbackData, setFeedbackData] = useState({
    question1: '',
    question2: '',
    textFeedback: '',
  });

  const handleNext = () => {
    if (step < 4) {
      setAnimateClass('slide-exit'); // Apply exit animation

      setTimeout(() => {
        setStep(step + 1);
        setAnimateClass('slide-enter'); // Apply enter animation
      }, 500); // Animation duration
    }

    if (step === 3) {
      setShowSubmit(true);
    }
  };

  const handleBack = () => {
    if (step > 1) {
      setAnimateClass('slide-exit'); // Apply exit animation

      setTimeout(() => {
        setStep(step - 1);
        setAnimateClass('slide-enter'); // Apply enter animation
      }, 500); // Animation duration
    }
  };

  const handleChange = (e) => {
    setFeedbackData({
      ...feedbackData,
      [e.target.name]: e.target.value,
    });
  };

  const handleButtonClick = (question, num) => {
    setFeedbackData((prevData) => ({
      ...prevData,
      [question]: num,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log(feedbackData);
    setSubmitted(true);
    setShowSubmit(false);
  };

  return (
    <div className="feedback-page">
      <div className={`feedback-container ${submitted ? 'slide-out' : ''}`}>
        {!submitted ? (
          <>
            <div className={`feedback-form ${showSubmit ? 'dimmed' : ''}`}>
              <div className="progress-bar">
                <span>{step}/4</span>
                <div className="bar">
                  <div className="progress" style={{ width: `${(step / 4) * 100}%` }}></div>
                </div>
              </div>

              <div className={`feedback-step ${animateClass}`}>
                {step === 1 && (
                  <div className="step-content">
                    <h2>How would you rate your overall experience using the roadmap feature?</h2>
                    <div className="rating-options">
                      {[1, 2, 3, 4, 5].map((num) => (
                        <button
                          key={num}
                          className={feedbackData.question1 === num ? 'active' : ''}
                          onClick={() => handleButtonClick('question1', num)}
                        >
                          {num}
                        </button>
                      ))}
                    </div>
                    <button onClick={handleNext}>Next</button>
                  </div>
                )}

                {step === 2 && (
                  <div className="step-content">
                    <h2>Did the roadmap help you track your progress effectively?</h2>
                    <div className="rating-options">
                      {[1, 2, 3, 4, 5].map((num) => (
                        <button
                          key={num}
                          className={feedbackData.question2 === num ? 'active' : ''}
                          onClick={() => handleButtonClick('question2', num)}
                        >
                          {num}
                        </button>
                      ))}
                    </div>
                    {/* <button className="back-arrow" onClick={handleBack}>{"<<"}</button> */}
                    <button onClick={handleNext}>Next</button>
                  </div>
                )}

                {step === 3 && (
                  <div className="step-content">
                    <h2>Please provide any suggestions or improvements for the roadmap feature.</h2>
                    <textarea
                      name="textFeedback"
                      placeholder="Write your feedback here..."
                      value={feedbackData.textFeedback}
                      onChange={handleChange}
                      required
                    />
                    {/* <button onClick={handleBack}>{"<<"}</button> */}
                    <button onClick={handleNext}>Next</button>
                  </div>
                )}
              </div>
            </div>

            {showSubmit && (
              <div className="submit-overlay">
                <div className="submit-content">
                  <h2>Submit your feedback</h2>
                  <button onClick={handleSubmit}>Submit</button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="thank-you-message slide-in">
            <h2>Thank you for your feedback!</h2>
          </div>
        )}
      </div>
    </div>
  );
};

export default Feedback;