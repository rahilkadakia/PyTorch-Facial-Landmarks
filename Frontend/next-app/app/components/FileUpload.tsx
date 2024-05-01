'use client';
import React, { useState, ChangeEvent, useEffect } from 'react';
import DisplayImage from './DisplayImage';
import Image from 'next/image';

const FileUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fd, setFd] = useState<FormData | null>(null); // State to manage FormData object
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  selectedFile && console.log('FileIpload Component SelectedFile @@@@@@@@@@@');
  useEffect(() => {
    const processImage = async () => {
    try {
      const res = await fetch('http://dynamic-nomad-414417.ue.r.appspot.com/process-image', {
        method: 'POST',
        body: fd,
      });
      const blob = await res.blob(); // get the image as a blob
      const url = URL.createObjectURL(blob); // create a URL from the blob
      setImageUrl(url); // save the URL in the state
    } catch (error) {
      console.error('Error processing image:', error);
    }
  };
    fd && processImage(); // initiate the process when the component mounts
  }, [fd]);
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      setSelectedFile(file);

      // Create a URL from the file to use as the src for the preview image
      // const url = URL.createObjectURL(file);
      // setPreviewUrl(url);

      // Reset fd whenever a new file is uploaded
       setImageUrl(null);
    }
  };

  const handleClick = () => {
    if (selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);
      setFd(formData); 
      // Update state with the FormData object
    } else {
      console.log('No file selected.');
    }
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-gray-900 text-white">
      <header className="w-full py-8 bg-gray-800 text-center text-4xl font-bold text-white shadow-md mb-16">
        Facial Landmark Detection
      </header>
      <div className="card flex flex-col items-center w-full md:w-3/4 lg:w-1/2 xl:w-3/5 2xl:w-1/2 bg-gray-800 shadow-xl rounded-lg overflow-hidden mx-5">
        <div className="p-6 flex flex-col gap-4 items-center">
          <h2 className="text-2xl font-bold mb-2 text-white">Upload Image</h2>
          <div className="flex justify-between w-full">
            <button className="btn btn-outline btn-warning w-full md:w-2/5" onClick={() => document.getElementById('image-input')?.click()}>Upload Image To Be Processed </button>
            <input style={{ visibility: 'hidden' }} type="file" className="file-input file-input-bordered file-input-primary w-full md:w-2/5" id="image-input" accept="image/*" onChange={handleFileChange} />
            <button className="btn btn-outline btn-success w-full md:w-2/5" onClick={handleClick}>Generate Landmarks</button>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4 p-6">
          <figure className=""> 
          <div style={{ width: '300px', height: '400px' }}> {/* Set custom dimensions */}
         {selectedFile && <Image src={URL.createObjectURL(selectedFile)} alt="Preview" layout='responsive' width={300} height={400}/>}
        </div>
           </figure>

          <figure className="">          {imageUrl && <DisplayImage imageUrl={imageUrl} />} </figure>
        </div>
      </div>
      <footer className="w-full py-8 bg-gray-800 text-center text-lg font-semibold text-white shadow-md mt-16">
        © 2024 Facial Landmark Detection
      </footer>
    </div>
  );
};

export default FileUpload;
