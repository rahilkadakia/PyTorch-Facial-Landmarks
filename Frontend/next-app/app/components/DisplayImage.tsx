import React from 'react';

interface DisplayImageProps {
  fd: FormData;
}

const DisplayImage: React.FC<DisplayImageProps> = ({ fd }) => {
  const processImage = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/process-image', {
        method: 'POST',
        body: fd,
      });
      const image = await res.json(); // assuming the response is JSON
      console.log(image); // handle the response as needed
    } catch (error) {
      console.error('Error processing image:', error);
    }
  };

  processImage(); // initiate the process when the component mounts

  return (
    <div>
      <p>Processing image...</p>
    </div>
  );
};

export default DisplayImage;
