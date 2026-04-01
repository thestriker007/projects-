import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useParams, useNavigate } from 'react-router-dom';
import './App.css';

// --- MOCK DATABASE (INSECURE) ---
const mockDb = {
  users: [{ id: 1, username: 'admin', password: 'password123', role: 'admin' }],
  products: [{ id: 101, name: 'Fancy Gadget', description: 'A very cool gadget.', price: 99.99, reviews: [] }],
};

// --- MOCK API ENDPOINTS (INSECURE) ---
// This is not a real server, just a function to simulate server-side logic
const mockApi = {
  login: (username, password) => {
    // VULNERABILITY: SQL INJECTION (SIMULATED)
    // In a real backend, this would be a vulnerable SQL query.
    // For this mock, we'll just show the logic.
    const sqlQuery = `SELECT * FROM users WHERE username = '${username}' AND password = '${password}'`;
    console.log(`Simulated SQL Query: ${sqlQuery}`);

    // This is the broken logic. The real vulnerability is on the backend.
    if (username === "' OR '1'='1" && password === "' OR '1'='1") {
      return { success: true, user: mockDb.users[0] }; // Bypasses authentication
    }
    const user = mockDb.users.find(u => u.username === username && u.password === password);
    return user ? { success: true, user } : { success: false, message: 'Invalid credentials' };
  },

  getProduct: (id) => {
    const product = mockDb.products.find(p => p.id === parseInt(id));
    return product ? { success: true, product } : { success: false, message: 'Product not found' };
  },

  addReview: (productId, reviewContent) => {
    const product = mockDb.products.find(p => p.id === parseInt(productId));
    if (product) {
      // VULNERABILITY: STORED XSS - No sanitization
      product.reviews.push({ content: reviewContent });
      return { success: true, reviews: product.reviews };
    }
    return { success: false, message: 'Product not found' };
  },

  deleteProduct: (id, userRole) => {
    // VULNERABILITY: BROKEN ACCESS CONTROL
    // The check is only on the frontend, not the backend (simulated here)
    const initialLength = mockDb.products.length;
    mockDb.products = mockDb.products.filter(p => p.id !== parseInt(id));
    return mockDb.products.length < initialLength ? { success: true } : { success: false, message: 'Product not found' };
  },
};

// --- COMPONENTS ---

const Header = ({ user, setUser }) => {
  const navigate = useNavigate();
  const handleLogout = () => {
    setUser(null);
    navigate('/');
  };
  return (
    <header className="App-header">
      <nav>
        <Link to="/">Home</Link>
        <Link to="/search">Search</Link>
        {user ? (
          <>
            <Link to="/profile">Profile</Link>
            {user.role === 'admin' && <Link to="/admin">Admin</Link>}
            <button onClick={handleLogout}>Logout</button>
          </>
        ) : (
          <Link to="/login">Login</Link>
        )}
      </nav>
    </header>
  );
};

const HomePage = () => (
  <div>
    <h2>Welcome to VulnStack E-commerce</h2>
    <p>Check out our featured products below!</p>
  </div>
);

const SearchPage = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const navigate = useNavigate();

  const handleSearch = () => {
    // VULNERABILITY: REFLECTED XSS
    // The query is directly put into the URL.
    navigate(`/search?q=${query}`);
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const q = params.get('q');
    if (q) {
      setQuery(q);
      // In a real app, this would query a database. Here we just mock it.
      setResults(mockDb.products.filter(p => p.name.includes(q)));
    }
  }, [window.location.search]);

  return (
    <div>
      <h2>Search Products</h2>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter search term"
      />
      <button onClick={handleSearch}>Search</button>
      {/* VULNERABILITY: REFLECTED XSS */}
      {query && <p>Search results for: <span dangerouslySetInnerHTML={{ __html: query }} /></p>}
      <ul>
        {results.map(p => (
          <li key={p.id}><Link to={`/product/${p.id}`}>{p.name}</Link></li>
        ))}
      </ul>
    </div>
  );
};

const ProductPage = () => {
  const { productId } = useParams();
  const [product, setProduct] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [newReview, setNewReview] = useState('');

  useEffect(() => {
    const result = mockApi.getProduct(productId);
    if (result.success) {
      setProduct(result.product);
      setReviews(result.product.reviews);
    }
  }, [productId]);

  const handleAddReview = () => {
    const result = mockApi.addReview(productId, newReview);
    if (result.success) {
      setReviews(result.reviews);
      setNewReview('');
    }
  };

  if (!product) return <div>Product not found!</div>;

  return (
    <div>
      <h2>{product.name}</h2>
      <p>{product.description}</p>
      <h3>Reviews</h3>
      <ul>
        {reviews.map((review, index) => (
          // VULNERABILITY: STORED XSS
          <li key={index} dangerouslySetInnerHTML={{ __html: review.content }} />
        ))}
      </ul>
      <div>
        <h4>Add a review</h4>
        <textarea
          value={newReview}
          onChange={(e) => setNewReview(e.target.value)}
        />
        <button onClick={handleAddReview}>Submit Review</button>
      </div>
    </div>
  );
};

const LoginPage = ({ setUser }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleLogin = () => {
    const result = mockApi.login(username, password);
    if (result.success) {
      setUser(result.user);
      navigate('/');
    } else {
      alert('Login failed!');
    }
  };

  return (
    <div>
      <h2>Login</h2>
      <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Username" />
      <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" />
      <button onClick={handleLogin}>Login</button>
    </div>
  );
};

const AdminPage = ({ user }) => {
  const navigate = useNavigate();

  // VULNERABILITY: BROKEN ACCESS CONTROL (Frontend-only check)
  useEffect(() => {
    if (!user || user.role !== 'admin') {
      navigate('/');
    }
  }, [user, navigate]);

  const handleDeleteProduct = (productId) => {
    // VULNERABILITY: CSRF & BAC
    // This action is a simple GET request.
    // The server-side should check user role, but in our mock, it doesn't.
    const result = mockApi.deleteProduct(productId, user.role);
    if (result.success) {
      alert(`Product ${productId} deleted!`);
      navigate('/'); // Redirect after deletion
    } else {
      alert('Failed to delete product.');
    }
  };

  return (
    <div>
      <h2>Admin Panel</h2>
      <p>This is a secret area for admins only.</p>
      <button onClick={() => handleDeleteProduct(101)}>Delete Product (ID 101)</button>
    </div>
  );
};

function App() {
  const [user, setUser] = useState(null);

  return (
    <Router>
      <div className="App">
        <Header user={user} setUser={setUser} />
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/product/:productId" element={<ProductPage />} />
            <Route path="/login" element={<LoginPage setUser={setUser} />} />
            <Route path="/admin" element={<AdminPage user={user} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
