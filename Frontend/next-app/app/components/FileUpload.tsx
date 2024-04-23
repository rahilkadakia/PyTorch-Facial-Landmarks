'use client';
import React, { useState, ChangeEvent } from 'react';
import DisplayImage from './DisplayImage';

const FileUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fd, setFd] = useState<FormData | null>(null); // State to manage FormData object
  const [previewUrl, setPreviewUrl] = useState<string | null>(null); // State to manage preview URL

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      setSelectedFile(file);

      // Create a URL from the file to use as the src for the preview image
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
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
      {previewUrl && <img src={previewUrl} alt="Preview" height={500} width={500} />} {/* Display the preview image if the URL exists */}
      {fd && <DisplayImage fd={fd} />} {/* Pass fd to DisplayImage if it exists */}
    </>
  );
};

export default FileUpload;
