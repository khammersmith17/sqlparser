/*
#[allow(dead_code)]
pub enum SqlOp {
    GreaterThan,
    LessThan,
    EqaulTo,
    GreaterThanEqualTo,
    LessThanEqualTo,
}

#[allow(dead_code)]
pub enum Column {
    Star,
    Name(Vec<String>),
}

// recursively traverse thee operations and apply the filtering
#[allow(dead_code)]
pub enum SqlCondition {
    Number(i64),
    Identifier(String),
    Operation {
        left: Box<SqlCondition>,
        op: SqlOp,
        right: Box<SqlCondition>,
    },
}

#[allow(dead_code)]
pub struct SqlStatement {
    columns: Column,
    table: String,
    conditions: Vec<SqlCondition>,
}
*/
#[derive(Debug)]
pub struct BasicSelectStatement<'a> {
    columns: Vec<&'a str>,
    table: &'a str,
}
// this is returned from a parser function
// it returns the part that has been parsed out and then the remaining portion of the text/tokens
type ParseResult<'a, Output> = Result<(&'a str, Output), String>;

// a trait with the function parse which takes in a str and returns some parsed str and additional
// tokens in output
trait Parser<'a, Output> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, Output>;
}

// a container that holds a function pointer to a boxed parser function
struct BoxedParser<'a, Output> {
    f: Box<dyn Fn(&'a str) -> ParseResult<'a, Output> + 'a>,
}

// accept a parser function and box it
impl<'a, Output> BoxedParser<'a, Output> {
    fn new<F>(f: F) -> BoxedParser<'a, Output>
    where
        F: Fn(&'a str) -> ParseResult<'a, Output> + 'a,
    {
        BoxedParser { f: Box::new(f) }
    }
}

// take the boxed parser function and parse the input
impl<'a, Output> Parser<'a, Output> for BoxedParser<'a, Output> {
    fn parse(&self, input: &'a str) -> ParseResult<'a, Output> {
        // need to wrap f in () because it is a field with a function pointer and not a method on
        // the struct
        // in order to just for self.f(), f needs to be a method on the struct
        // in this case it is a function pointer in a field and not a function
        (self.f)(input)
    }
}

trait ParserExt<'a, Output>: Parser<'a, Output> + Sized {
    fn then<P2, Output2>(self, next: P2) -> BoxedParser<'a, (Output, Output2)>
    where
        P2: Parser<'a, Output2> + 'a,
        Output2: 'a,
        Output: 'a,
        Self: 'a,
    {
        BoxedParser::new(move |input| {
            self.parse(input).and_then(|(next_input, result)| {
                next.parse(next_input)
                    .map(|(final_input, next_result)| (final_input, (result, next_result)))
            })
        })
    }

    fn map<F, Output2>(self, f: F) -> BoxedParser<'a, Output2>
    where
        F: Fn(Output) -> Output2 + 'a,
        Output: 'a,
        Output2: 'a,
        Self: 'a,
    {
        BoxedParser::new(move |input| self.parse(input).map(|(rest, val)| (rest, f(val))))
    }
}

impl<'a, Output, P> ParserExt<'a, Output> for P where P: Parser<'a, Output> + Sized {}

// then define functions to return a BoxedParser
// define a closure then move that closure into a boxed parser
fn digit_char<'a>() -> impl Parser<'a, char> {
    BoxedParser::new(|input| {
        let mut chars = input.chars();
        if let Some(first) = chars.next() {
            if first.is_ascii_digit() {
                let rest = &input[first.len_utf8()..];
                return Ok((rest, first));
            }
        }
        Err("Expected digit".into())
    })
}

fn multi_digit_char<'a, Output, P>(parser: P) -> impl Parser<'a, Vec<Output>>
where
    P: Parser<'a, Output> + 'a,
    Output: 'a,
{
    BoxedParser::new(move |mut input| {
        let mut res: Vec<Output> = Vec::new();

        while let Ok((next, val)) = parser.parse(input) {
            res.push(val);
            input = next;
        }
        Ok((input, res))
    })
}

fn parse_for_select<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let keyword = "select";

        let (head, rest) = input.split_at(keyword.len());

        if head.eq_ignore_ascii_case(head) {
            Ok((rest, keyword))
        } else {
            Err("Invalid query syntax".into())
        }
    })
}

fn parse_for_from<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let keyword = "from";

        let (head, rest) = input.split_at(keyword.len());

        if head.eq_ignore_ascii_case(head) {
            Ok((rest, keyword))
        } else {
            Err("Invalid query syntax".into())
        }
    })
}

// parse until we reach then end of a word
fn parse_for_identifiers<'a>() -> impl Parser<'a, &'a str> {
    BoxedParser::new(|input| {
        let mut end = 0_usize;

        for c in input.chars() {
            if c.is_ascii_alphanumeric() || c == '_' {
                end += 1
            } else {
                break;
            }
        }

        if end == 0 {
            println!("chars[0]: {:?}", input);
            Err("expected identifier".into())
        } else {
            Ok((&input[end..], &input[..end]))
        }
    })
}

// parse until we reach a space
// then return the remaining input and an empty ()
fn parse_for_whitespace<'a>() -> impl Parser<'a, ()> {
    BoxedParser::new(|input| {
        let len = input.len();
        let mut end = 0_usize;
        for c in input.bytes() {
            if c == 32 {
                end += 1
            } else {
                break;
            }
        }

        if end == len {
            Err("Expected a space".into())
        } else {
            Ok((&input[end..], ()))
        }
    })
}

fn sep_by_two_seperators<'a, P, S, W, Output>(
    item: P,
    whitespace: W, // seperator on whitespace
    seperator: S,  // seperator on comma
) -> impl Parser<'a, Vec<Output>>
where
    P: Parser<'a, Output> + 'a,
    S: Parser<'a, ()> + 'a,
    W: Parser<'a, ()> + 'a,
    Output: 'a,
{
    BoxedParser::new(move |mut input| {
        let mut res_items = Vec::new();

        // parse out the first identifier from the input
        match item.parse(input) {
            Ok((next, res)) => {
                res_items.push(res);
                input = next;
            }
            Err(_) => return Ok((input, res_items)),
        }

        // parse for seperator
        // then parse for identifier
        // until it is exhausted
        while let Ok((next, _)) = seperator.parse(input) {
            input = next;

            match whitespace.parse(input) {
                Ok((next, _)) => {
                    input = next;
                }
                Err(_) => {}
            };
            match item.parse(input) {
                Ok((next, res)) => {
                    res_items.push(res);
                    input = next;
                }
                Err(e) => return Err(e),
            }
        }
        Ok((input, res_items))
    })
}

fn parse_for_columns<'a>() -> impl Parser<'a, Vec<&'a str>> {
    sep_by_two_seperators(
        parse_for_identifiers(),
        parse_for_whitespace(),
        BoxedParser::new(|input| {
            if input.starts_with(",") {
                return Ok((&input[1..], ()));
            } else {
                return Err("Expected comma".into());
            }
        }),
    )
}

fn basic_select_statement<'a>() -> impl Parser<'a, BasicSelectStatement<'a>> {
    parse_for_select()
        .then(parse_for_whitespace())
        .then(parse_for_columns())
        .then(parse_for_whitespace())
        .then(parse_for_from())
        .then(parse_for_whitespace())
        .then(parse_for_identifiers())
        .map(|((((((_, _), columns), _), _), _), table)| BasicSelectStatement { columns, table })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_single() {
        let parser = digit_char();
        let res = parser.parse("1122aa");
        println!("{:?}", res);
        assert_eq!(res, Ok(("122aa", '1')))
    }

    #[test]
    fn test_multiple() {
        let digit_parser = digit_char();
        let mdp = multi_digit_char(digit_parser);
        let res = mdp.parse("11122aa");
        println!("{:?}", res);
        assert_eq!(res, Ok(("aa", vec!['1', '1', '1', '2', '2'])))
    }

    #[test]
    fn select_parser() {
        let q = "Select * from table";
        let parser = parse_for_select();
        let res = parser.parse(q);
        println!("{:?}", res);
    }

    #[test]
    fn parser_for_identifier_and_whitespace() {
        let parser = parse_for_identifiers();
        let res = parser.parse("word1 word2").unwrap();

        let wparser = parse_for_whitespace();

        let res2 = wparser.parse(res.0);
        println!("{:?}", res2);
    }

    /*
    #[test]
    fn test_sep_by() {
        let input = "word word word";
        let id_parser = parse_for_identifiers();
        let wparser = parse_for_whitespace();
        let parser = sep_by(id_parser, wparser);
        let res = parser.parse(input);

        println!("{:?}", res);
    }

    */
    #[test]
    fn test_chain() {
        let chain = parse_for_select()
            .then(parse_for_whitespace())
            .then(parse_for_columns())
            .then(parse_for_whitespace())
            .then(parse_for_from())
            .then(parse_for_whitespace())
            .then(parse_for_identifiers())
            .map(|input| println!("input: {:?}", input));
        let _ = chain.parse("select column from table");
    }

    #[test]
    fn test_whole_parser() {
        let parser = basic_select_statement();
        let res = parser.parse("select column from table").unwrap();

        println!("res: {:?}", res);
    }

    #[test]
    fn multiple_columns() {
        let parser = basic_select_statement();
        let res = parser.parse("select column1, column2 from table").unwrap();

        println!("res: {:?}", res);
    }
}
