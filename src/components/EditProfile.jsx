import React, { useState, useEffect } from 'react';
import './EditProfile.css';
import { useNavigate } from 'react-router-dom';

function EditProfile() {
 const [formData, setFormData] = useState({
  username: '',
  firstName: '',
  lastName: '',
  gender: '',
  location: '',
  email: '',
  phone: '',
  birthdate: '',
  profilePicture: null, // To store the image file
 });

 const navigate = useNavigate(); // Ensure navigation works

 useEffect(() => {
  // Fetch user profile data on component load using fetch
  const fetchProfile = async () => {
   try {
    const token = localStorage.getItem('token');

    if (!token) {
     navigate('/login');
     return;
    }

    const res = await fetch('http://localhost:5000/api/profile/me', {
     method: 'GET',
     headers: {
      Authorization: `Bearer ${token}`,
     },
    });

    if (!res.ok) {
     throw new Error('Failed to fetch profile data');
    }

    const data = await res.json();
    setFormData(data); // Set the form fields with user data
   } catch (error) {
    console.error('Error fetching user profile', error);
   }
  };
  fetchProfile();
 }, [navigate]);

 const handleInputChange = (e) => {
  const { name, value } = e.target;
  setFormData((prevState) => ({
   ...prevState,
   [name]: value,
  }));
 };

 const handleImageChange = (e) => {
  setFormData((prevState) => ({
   ...prevState,
   profilePicture: e.target.files[0], // Set the image file
  }));
 };

 const handleSubmit = async (e) => {
  e.preventDefault();
  const formDataToSend = new FormData();
  formDataToSend.append('username', formData.username);
  formDataToSend.append('firstName', formData.firstName);
  formDataToSend.append('lastName', formData.lastName);
  formDataToSend.append('gender', formData.gender);
  formDataToSend.append('location', formData.location);
  formDataToSend.append('email', formData.email);
  formDataToSend.append('phone', formData.phone);
  formDataToSend.append('birthdate', formData.birthdate);

  // If the user has selected a new profile picture, append it to the form data
  if (formData.profilePicture) {
   formDataToSend.append('profilePicture', formData.profilePicture);
  }

  try {
   const token = localStorage.getItem('token');

   const res = await fetch('http://localhost:5000/api/profile/me', {
    method: 'PUT',
    headers: {
     Authorization: `Bearer ${token}`,
    },
    body: formDataToSend,
   });

   if (!res.ok) {
    throw new Error('Failed to update profile');
   }

   alert('Profile updated successfully');
  } catch (error) {
   console.error('Error updating profile', error);
   alert('Error updating profile');
  }
 };

 return (
  <div className="page-container">
   <div className="content-container">
    <div className="section">
     <div className="profile-picture">
      <img
       src={formData.profilePicture ? URL.createObjectURL(formData.profilePicture) : "https://cdn-icons-png.flaticon.com/512/147/147144.png"}
       alt="Profile"
      />
      <input
       type="file"
       accept="image/*"
       onChange={handleImageChange}
       className="upload-button"
      />
     </div>
    </div>

    <div className="account-details">
     <h3 style={{ marginBottom: '20px', color: '#333' }}>Account Details</h3>
     <form onSubmit={handleSubmit}>
      <div className="form-group">
       <input
        type="text"
        name="username"
        value={formData.username}
        onChange={handleInputChange}
        placeholder="Enter your username"
       />
      </div>
      <div className="form-group">
       <input
        type="text"
        name="firstName"
        value={formData.firstName}
        onChange={handleInputChange}
        placeholder="Enter your first name"
       />
      </div>
      <div className="form-group">
       <input
        type="text"
        name="lastName"
        value={formData.lastName}
        onChange={handleInputChange}
        placeholder="Enter your last name"
       />
      </div>
      <div className="form-group">
       <input
        type="text"
        name="gender"
        value={formData.gender}
        onChange={handleInputChange}
        placeholder="Enter your gender"
       />
      </div>
      <div className="form-group">
       <input
        type="text"
        name="location"
        value={formData.location}
        onChange={handleInputChange}
        placeholder="Enter your location"
       />
      </div>
      <div className="form-group">
       <input
        type="email"
        name="email"
        value={formData.email}
        onChange={handleInputChange}
        placeholder="Enter your email address"
       />
      </div>
      <div className="form-group">
       <input
        type="tel"
        name="phone"
        value={formData.phone}
        onChange={handleInputChange}
        placeholder="Enter your phone number"
       />
      </div>
      <div className="form-group">
       <input
        type="date"
        name="birthdate"
        value={formData.birthdate}
        onChange={handleInputChange}
       />
      </div>
      <button type="submit" className="save-button">
       Save changes
      </button>
     </form>
    </div>
   </div>
  </div>
 );
}

export default EditProfile;
