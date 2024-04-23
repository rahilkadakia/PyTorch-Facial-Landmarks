'use client';
import React, { useState, ChangeEvent } from 'react';
import DisplayImage from './DisplayImage';

const FileUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fd, setFd] = useState<FormData | null>(null); // State to manage FormData object

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleClick = () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);
      setFd(formData); // Update state with the FormData object
    } else {
      console.log('No file selected.');
    }
  };

  return (
    <>
      <label htmlFor="uploadFile">Upload File</label> <br />
      <input
        type="file"
        id="image-input"
        accept="image/*"
        onChange={handleFileChange}
      />
      <br />
      <button className="btn btn-primary" onClick={handleClick}>
        Upload
      </button>
      {fd && <DisplayImage fd={fd} />} {/* Pass fd to DisplayImage if it exists */}
    </>
  );
};

export default FileUpload;
