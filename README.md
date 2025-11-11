# Routing

A flexible and efficient routing library for building scalable applications.

## Overview

This project provides a robust routing solution designed to handle URL routing, request dispatching, and navigation management. Whether you're building a web application, API server, or any system that requires intelligent request routing, this library offers the tools you need.

## Features

- **Fast Route Matching**: Efficient algorithm for matching incoming requests to registered routes
- **Dynamic Parameters**: Support for path parameters and wildcards
- **Middleware Support**: Chain middleware functions for request processing
- **HTTP Method Handling**: Support for GET, POST, PUT, DELETE, PATCH, and custom methods
- **Nested Routes**: Organize routes hierarchically for better code structure
- **Route Groups**: Group related routes with shared prefixes and middleware
- **Query Parameter Parsing**: Built-in support for parsing query strings
- **Flexible Configuration**: Customize routing behavior to fit your needs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/routing.git
cd routing

# Install dependencies (if applicable)
npm install
```

## Quick Start

```javascript
// Basic usage example
const router = new Router();

// Define routes
router.get('/users', (req, res) => {
  res.json({ users: [] });
});

router.get('/users/:id', (req, res) => {
  const userId = req.params.id;
  res.json({ user: { id: userId } });
});

router.post('/users', (req, res) => {
  // Create new user
  res.status(201).json({ message: 'User created' });
});

// Start routing
router.listen(3000);
```

## Usage

### Defining Routes

Routes can be defined using HTTP method helpers:

```javascript
router.get('/path', handler);
router.post('/path', handler);
router.put('/path', handler);
router.delete('/path', handler);
router.patch('/path', handler);
```

### Path Parameters

Capture dynamic segments in your routes:

```javascript
router.get('/posts/:postId/comments/:commentId', (req, res) => {
  const { postId, commentId } = req.params;
  // Handle request
});
```

### Middleware

Add middleware to process requests before they reach route handlers:

```javascript
// Global middleware
router.use(logger);
router.use(authenticate);

// Route-specific middleware
router.get('/admin', authorize, adminHandler);
```

### Route Groups

Organize related routes together:

```javascript
router.group('/api/v1', (group) => {
  group.get('/users', getUsersHandler);
  group.post('/users', createUserHandler);
  group.get('/users/:id', getUserHandler);
});
```

## Configuration

Customize router behavior with configuration options:

```javascript
const router = new Router({
  strict: false,        // Enable strict route matching
  caseSensitive: false, // Case-sensitive routes
  mergeParams: true     // Merge parameters from parent routes
});
```

## API Documentation

### `Router(options)`

Creates a new router instance.

**Parameters:**
- `options` (Object): Configuration options

### `router.get(path, ...handlers)`

Register a GET route.

### `router.post(path, ...handlers)`

Register a POST route.

### `router.use(middleware)`

Add global middleware.

### `router.group(prefix, callback)`

Create a route group with a common prefix.

## Examples

Check the `/examples` directory for more detailed usage examples:

- Basic routing
- RESTful API
- Nested routes
- Middleware chains
- Error handling

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

## Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Run linter
npm run lint
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions:

- Open an issue on [GitHub Issues](https://github.com/yourusername/routing/issues)
- Check existing issues for solutions
- Consult the documentation

## Roadmap

- [ ] Add support for route aliasing
- [ ] Implement route caching for improved performance
- [ ] Add TypeScript definitions
- [ ] Support for async middleware
- [ ] Enhanced error handling and debugging tools

## Acknowledgments

Thanks to all contributors who have helped make this project better!

---

**Note**: This project is under active development. APIs may change before reaching v1.0.0.
