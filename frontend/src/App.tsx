import { Landing } from './pages/Landing'
import { Studio } from './pages/Studio'
import { useRoute } from './lib/router'

export default function App() {
  const route = useRoute()
  return route === 'studio' ? <Studio /> : <Landing />
}
