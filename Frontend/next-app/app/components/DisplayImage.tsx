import { useState, useEffect } from 'react';
import Image from 'next/image';
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
  <>
      {imageUrl ? (
        <div style={{ width: '400px', height: '400px' }}> {/* Set custom dimensions */}
          <Image src={imageUrl} alt="Processed" layout="responsive" width={400} height={400} />
        </div>
      ) : (
        <p>Processing image...</p>
      )}
    </>
  );
};

export default DisplayImage;
