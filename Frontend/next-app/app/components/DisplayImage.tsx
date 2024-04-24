import { useState, useEffect } from 'react';

interface DisplayImageProps {
  fd: FormData;
}

const DisplayImage: React.FC<DisplayImageProps> = ({ fd }) => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  const processImage = async () => {
    try {
      const res = await fetch('https://dynamic-nomad-414417.ue.r.appspot.com/process-image', {
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

  useEffect(() => {
    processImage(); // initiate the process when the component mounts
  }, []);

  return (
    <div>
      {imageUrl ? <img src={imageUrl} alt="Processed" /> : <p>Processing image...</p>}
    </div>
  );
};

export default DisplayImage;
