'use client';
import React, { useState, ChangeEvent } from 'react';

const FileUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleClick = () => {
    if (selectedFile) {
      let fd = new FormData();
      fd.append('file', selectedFile);
      // Use fd for further processing like making a POST request
      console.log(selectedFile);
      fd.forEach((value, key) => console.log(key, value));
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
    </>
  );
};

export default FileUpload;
