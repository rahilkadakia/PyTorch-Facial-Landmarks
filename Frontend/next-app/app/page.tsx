import Image from 'next/image'
import Link from 'next/link'
import ProductCard from './components/ProductCard'
import FileUpload from './components/FileUpload'

export default function Home(){
  return(
    <main>
      {/* <h1>Hellow Word</h1> */}
    {/* <Link href="/users">Users</Link>
    <ProductCard /> */}
    <FileUpload />
      </main>
  )
}
