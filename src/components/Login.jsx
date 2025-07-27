import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './Login.css';

function Login() {
    const [formData, setFormData] = useState({
        email: '',
        password: '',
    });

    const [errors, setErrors] = useState({});
    const navigate = useNavigate();

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value,
        });
    };

    const validateForm = () => {
        let errors = {};
        if (!formData.email) {
            errors.email = 'Email is required';
        } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
            errors.email = 'Email is invalid';
        }
        if (!formData.password) {
            errors.password = 'Password is required';
        }
        setErrors(errors);
        return Object.keys(errors).length === 0;
    };

    
    const handleSubmit = async (e) => {
       e.preventDefault();
      
       if (validateForm()) {
      try {
       const response = await fetch('http://localhost:5000/api/users/login', {
       method: 'POST',
       headers: {
        'Content-Type': 'application/json',
       },
       body: JSON.stringify(formData),
       });
      
       const data = await response.json(); // Parse the JSON response
       console.log(data); // Log the response for debugging
      
       if (data.authentication) { // Check for authentication
       localStorage.setItem('token', data.token);
       navigate('/dashboard');
       } else {
       setErrors({ apiError: 'Invalid credentials. Please try again.' });
       }
      } catch (error) {
       console.error('Login error:', error);
       setErrors({ apiError: 'Something went wrong. Please try again.' });
      }
       }
      };
      

    return (
        <div className="login-page">
            <div className="login-container">
                <div className="login-image">
                    <div className="welcome-message">Welcome</div>
                </div>
                <form className="login-form" onSubmit={handleSubmit}>
                    <h2 className="logIn">Log in</h2>
                    <div className="form-group">
                        <label>Email</label>
                        <input
                            type="email"
                            name="email"
                            value={formData.email}
                            onChange={handleChange}
                            required
                        />
                        {errors.email && <p className="error">{errors.email}</p>}
                    </div>
                    <div className="form-group">
                        <label>Password</label>
                        <input
                            type="password"
                            name="password"
                            value={formData.password}
                            onChange={handleChange}
                            required
                        />
                        {errors.password && <p className="error">{errors.password}</p>}
                    </div>
                    {errors.apiError && <p className="error">{errors.apiError}</p>}
                    <center>
                        <button type="submit">Login</button>
                    </center>
                    <p className="register-link">
                    </p>
                </form>
            </div>
        </div>
    );
}

export default Login;