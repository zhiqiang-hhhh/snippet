class Singleton {
    private:
        Singleton() {}
    public:
        static Singleton instance;
        static Singleton& getInstance() {
            return instance;
        }
};