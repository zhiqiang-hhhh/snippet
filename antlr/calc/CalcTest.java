import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.tree.ParseTree;

public class CalcTest {
    public static void main(String[] args) throws Exception {
        CalcLexer lexer = new CalcLexer(CharStreams.fromString("1 + 2 * (3 + 4) - 5 / 6"));
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        CalcParser parser = new CalcParser(tokens);
        ParseTree tree = parser.calc();
        System.out.println(tree.toStringTree(parser));
    }
}